# # step1
# from transformers import pipeline

# # step2
# classifier = pipeline("image-classification", model="rizvandwiki/gender-classification")

# # step3

# image = "k3.jpg"

# # step4
# result = classifier(image)

# # step5

# print(result)

import PIL.Image
from fastapi import FastAPI, File, UploadFile
from transformers import pipeline
import io
import numpy as np

# Initialize the classifier
classifier = pipeline("image-classification", model="dima806/man_woman_face_image_detection")

app = FastAPI()

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):
    try:
        # Read the uploaded file
        byte_file = await file.read()

        # Convert byte array to binary stream
        image_bin = io.BytesIO(byte_file)

        # Try to open the image using PIL
        try:
            pil_img = PIL.Image.open(image_bin)
        except PIL.UnidentifiedImageError:
            return {"error": "Cannot identify image file. Please upload a valid image."}

        # Classification
        result = classifier(pil_img)  # Use the PIL image directly

        return {"result": result}
    except Exception as e:
        return {"error": str(e)}