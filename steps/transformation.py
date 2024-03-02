from torchvision import transforms
from zenml import step


@step
def define_transformation() -> dict:
    """Define the image transformation pipeline."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return {'transform': transform}
