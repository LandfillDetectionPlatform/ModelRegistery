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

@step
def evaluate_model(model: torch.nn.Module, loaders: dict) -> dict:
    val_loader = loaders['val_loader']
    model.eval()
    all_preds, all_labels = [], []
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    precision = precision_score(all_labels, all_preds, average='binary', zero_division=1)
    recall = recall_score(all_labels, all_preds, average='binary', zero_division=1)
    return {"precision": precision, "recall": recall}