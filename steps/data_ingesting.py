from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from zenml import step

@step
def load_data(data_dir: str, test_size: float = 0.2, transformation: dict = None) -> dict:
    """Loads data and prepares DataLoader for training, validation, or processes an input image for inference."""
    if transformation is None:
        raise ValueError("Transformation dictionary is required.")

    # Load dataset for training and validation
    transform = transformation['transform']
    full_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    total_size = len(full_dataset)
    train_size = int((1 - test_size) * total_size)
    val_size = total_size - train_size

    # Splitting the dataset into train and validation sets
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    return {'train_loader': train_loader, 'val_loader': val_loader}
