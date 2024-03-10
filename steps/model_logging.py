import mlflow
import os
from zenml import step

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

@step(enable_cache=False)
def log_and_save(model: dict, metrics: dict, promotion_decision: bool, model_path: str):
    mlflow.log_metrics(metrics)
    model_container = model["model"]
    if promotion_decision:
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(model_container.state_dict(), os.path.join(model_path, 'model.pth'))
        print("Model saved.")
    else:
        print("Model not promoted.")