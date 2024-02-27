import os
import copy
import torch
from torch.optim import lr_scheduler
from torch.optim import optim
from torchvision import models, transforms
import mlflow
import mlflow.pytorch
from torch.utils.data import DataLoader
from dataset import CustomDataset  # Replace 'CustomDataset' with your dataset class
from sklearn.metrics import precision_score, recall_score, accuracy_score

# Set the tracking URI
mlflow.set_tracking_uri("http://localhost:5000")  # replace with your tracking URI

# Function to train the model
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=6):
    # Define the path to save the model checkpoints
    model_save_dir = "../models"
    os.makedirs(model_save_dir, exist_ok=True)

    # Early stopping parameters
    early_stopping_patience = 10
    epochs_since_improvement = 0
    best_loss = float('inf')
    best_model_wts = None

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = val_loader

            running_loss = 0.0
            all_labels = []
            all_predictions = []

            for inputs, labels in dataloader:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    # Backward and optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item()

                # Collect predictions for metrics
                _, predictions = torch.max(outputs, 1)
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predictions.cpu().numpy())

            epoch_loss = running_loss / len(dataloader)
            print(f'{phase} Loss: {epoch_loss:.4f}')

            # Calculate metrics
            precision = precision_score(all_labels, all_predictions)
            recall = recall_score(all_labels, all_predictions)
            accuracy = accuracy_score(all_labels, all_predictions)

            # Log metrics to MLflow
            mlflow.log_metric(f'{phase}_loss', epoch_loss)
            mlflow.log_metric(f'{phase}_precision', precision)
            mlflow.log_metric(f'{phase}_recall', recall)
            mlflow.log_metric(f'{phase}_accuracy', accuracy)

            # Checkpointing
            if phase == 'val':
                checkpoint_path = os.path.join(model_save_dir, f'model_epoch_{epoch+1}.pth')
                torch.save(model.state_dict(), checkpoint_path)
                print(f"Checkpoint saved to {checkpoint_path}")

                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    epochs_since_improvement = 0
                else:
                    epochs_since_improvement += 1

                if epochs_since_improvement == early_stopping_patience:
                    print("Early stopping")
                    break

            if phase == 'train':
                scheduler.step()

    # Load best model weights
    model.load_state_dict(best_model_wts)

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load your dataset
# You need to replace 'CustomDataset' with your actual dataset class and provide the appropriate arguments
train_dataset = CustomDataset(root='path/to/train/dataset', version='v1', transform=transforms.Compose([...]))
val_dataset = CustomDataset(root='path/to/val/dataset', version='v1', transform=transforms.Compose([...]))

# Define your DataLoader
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2)

# Initialize the model
model = models.resnet50(pretrained=False)
num_classes = 2  # Legal and Illegal landfills
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

# Define the loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)  # Adjust these parameters as needed
num_epochs = 20
# Train the model and log metrics to MLflow
with mlflow.start_run() as run:
    # Log parameters
    mlflow.log_param("num_epochs", num_epochs)
    mlflow.log_param("learning_rate", 0.001)
    mlflow.log_param("batch_size", 4)
    mlflow.log_param("early_stopping_patience", early_stopping_patience)

    # Log dataset version
    mlflow.log_param("dataset_version", "v1")  # Replace with the actual version

    # Train the model
    train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs)
