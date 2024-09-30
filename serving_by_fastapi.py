import os
import io
from enum import Enum

import cv2
import cvlib as cv
from cvlib.object_detection import draw_bbox
import uvicorn
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse

# Create FastAPI instance
app = FastAPI(title='Deploying an ML Model with FastAPI')


class Model(str, Enum):
    yolov3tiny = "yolov3-tiny"
    yolov3 = "yolov3"


def create_output_dir(dir_name="images_uploaded"):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)


def validate_image(filename: str) -> bool:
    return filename.split(".")[-1] in ("jpg", "jpeg", "png")


def read_image_file(file: UploadFile) -> np.ndarray:
    # Read image as a stream of bytes
    image_stream = io.BytesIO(file.file.read())

    # Start the stream from the beginning (position zero)
    image_stream.seek(0)
    file_bytes = np.asarray(bytearray(image_stream.read()), dtype=np.uint8)
    return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)


# Run object detection
def detect_objects(image: np.ndarray, model: Model) -> tuple:
    return cv.detect_common_objects(image, model=model)


def create_output_image(image: np.ndarray, bbox: list, label: list, conf: list) -> np.ndarray:
    return draw_bbox(image, bbox, label, conf)


def save_image(image: np.ndarray, filename: str) -> None:
    cv2.imwrite(f'images_uploaded/{filename}', image)


# By using @app.get("/") you are allowing the GET method to work for the / endpoint.
@app.get("/")
def home():
    return "Congratulations! Your API is working as expected. Now head over to http://docs"


# This endpoint handles all the logic necessary for the object detection to work.
# It requires the desired model and the image in which to perform object detection.
@app.post("/predict")
def prediction(model: Model, file: UploadFile = File(...)):
    # 1. VALIDATE INPUT FILE
    if not validate_image(file.filename):
        raise HTTPException(status_code=415, detail="Unsupported file provided.")

    # 2. TRANSFORM RAW IMAGE INTO CV2 image
    image = read_image_file(file)

    # 3. RUN OBJECT DETECTION MODEL
    bbox, label, conf = detect_objects(image, model)

    # Create image that includes bounding boxes and labels
    output_image = create_output_image(image, bbox, label, conf)

    # Save it in a folder within the server
    save_image(output_image, file.filename)

    # 4. STREAM THE RESPONSE BACK TO THE CLIENT
    file_image = open(f'images_uploaded/{file.filename}', mode="rb")
    return StreamingResponse(file_image, media_type="image/jpeg")


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)
