from PIL import Image
from torchvision import transforms
from zenml import step
import torch

@step
def run_inference(model: torch.nn.Module, image_str: str, transformation: dict) -> dict:
    """Runs inference on the preprocessed image."""
    # Open the image
    image = Image.open(image_str)

    # Convert image to RGB if it's not already
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Apply the transformation
    transform = transformation['transform']
    input_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    # Ensure the model is in evaluation mode
    model.eval()

    # Move the tensor to the same device as the model
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)

    # Run inference
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)

    # Return the prediction
    return {'prediction': predicted.item()}