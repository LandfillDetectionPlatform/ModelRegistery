import os
import subprocess
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import StepLR
import mlflow
import mlflow.pytorch
from zenml import step
import dvc
import subprocess
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score

@step
def evaluate_model(model_data: dict, loaders: dict) -> dict:
    """
    Evaluate the model and log metrics and parameters to ZenML with MLflow integration.
    
    Args:
    - model: The PyTorch model to evaluate.
    - loaders: A dictionary containing the DataLoader for validation.
    - model_params: A dictionary of model parameters and hyperparameters for logging.
    
    Returns:
    A dictionary of evaluation metrics.
    """
    model = model_data['model']
    model_params = model_data['hyperparameters']

    val_loader = loaders['val_loader']
    model.eval()
    all_preds, all_labels = [], []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate additional metrics
    precision = precision_score(all_labels, all_preds, average='binary', zero_division=1)
    recall = recall_score(all_labels, all_preds, average='binary', zero_division=1)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='binary', zero_division=1)
    
    # Log metrics
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1_score", f1)

    # Log model parameters and hyperparameters
    for param, value in model_params.items():
        mlflow.log_param(param, value)

    # Optionally, log model architecture
    # mlflow.log_text(str(model), "model_architecture.txt")
    
    return {
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "f1_score": f1
    }
