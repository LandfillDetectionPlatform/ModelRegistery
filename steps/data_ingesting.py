from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from torchvision.transforms.functional import to_pil_image
from PIL import Image
import os
from zenml import step

@step
def load_data(data_dir: str, test_size: float = 0.2, inference: bool = False) -> dict:
    """Loads data and prepares DataLoader for training, validation, or processes an input image for inference."""
    # Common transformations
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    if inference:
        # Assuming data_dir points to a single image for inference
        image = Image.open(data_dir)
        image = transform(image)
        # Return a dictionary with the image tensor; no need for DataLoader
        return {'inference_image': image}
    else:
        # Load dataset for training and validation
        full_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
        total_size = len(full_dataset)
        train_size = int((1 - test_size) * total_size)
        val_size = total_size - train_size

        # Splitting the dataset into train and validation sets
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32)

        return {'train_loader': train_loader, 'val_loader': val_loader}