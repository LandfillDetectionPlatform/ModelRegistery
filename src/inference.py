from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from PIL import Image, UnidentifiedImageError
from torchvision import transforms
from model_utils import load_model, run_inference, save_preprocessed_image
import io
import logging
import os

app = FastAPI()
model = load_model()

# Specify the output directories
output_dir_1 = "../imagery/1"
output_dir_0 = "../imagery/0"

# Ensure the output directories exist, create them if not
os.makedirs(output_dir_1, exist_ok=True)
os.makedirs(output_dir_0, exist_ok=True)

@app.post("/classify-image/")
async def classify_image(file: UploadFile = File(...), name: str = Form(...)):
    try:
        # Read the uploaded image file
        contents = await file.read()

        # Convert the image bytes to a PIL Image
        pil_image = Image.open(io.BytesIO(contents))

        # Preprocess the image
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        input_tensor = transform(pil_image).unsqueeze(0)  # Add batch dimension

        # Run inference
        result = run_inference(model, input_tensor)

        # Save the preprocessed image to the output folder
        if int(result) == 1:
            saved_image_path = save_preprocessed_image(contents,name, output_dir_1)
        else:
            saved_image_path = save_preprocessed_image(contents,name, output_dir_0)

        # Display the result

        return {"result": result, "saved_image_path": saved_image_path}

    except UnidentifiedImageError as e:
        error_detail = "Invalid image file. Please upload a valid image."
        logging.error(f"Error in API endpoint: {error_detail}")
        raise HTTPException(status_code=400, detail=error_detail)

    except Exception as e:
        logging.error(f"Error in API endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))
