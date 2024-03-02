from zenml import step
from PIL import Image
import os

@step
def save_image_based_on_prediction(image_str: str, prediction: dict, save_dir: str) -> str:
    """Save the image to a folder based on the prediction."""
    target_path = os.path.join(save_dir, str(prediction['prediction']))
    image = Image.open(image_str)
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    print(prediction)
    image_name = os.path.basename(image.filename)
    image_path = os.path.join(target_path, image_name)
    image.save(image_path)
    return f"Image saved to {target_path}"
