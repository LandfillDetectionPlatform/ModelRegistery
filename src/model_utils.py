import torch
from torchvision import models, transforms
import io
import os
from PIL import Image
def load_model():
    model = models.resnet50(pretrained=False)
    num_classes = 2  # Legal and Illegal landfills
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    state_dict = torch.load('../models/illegal_landfills_model.pth', map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)

    model.eval()
    return model


def preprocess_image(image_path):
    # Load the image
    pil_image = Image.open(image_path)

    # Define the transformation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Preprocess the image
    input_tensor = transform(pil_image).unsqueeze(0)  # Add batch dimension
    return input_tensor

def run_inference(model, input_tensor):
    # Run inference
    with torch.no_grad():
        model.eval()
        outputs = model(input_tensor)
        _, predicted_class = torch.max(outputs, 1)
    return predicted_class.item()



def save_preprocessed_image(image_data, name, output_dir):
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Convert the image bytes to a PIL Image
    pil_image = Image.open(io.BytesIO(image_data))

    # Save the image with the given name in the output directory
    saved_image_path = os.path.join(output_dir, name)
    # Assuming you want to save the image in PNG format
    pil_image.save(saved_image_path)

    return saved_image_path
