import os
import torch
import torch.nn as nn
from torchvision import  models
from zenml import step

@step
def initialize_model(model_url: str) -> torch.nn.Module:
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 2)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    if os.path.exists(model_url):
        model.load_state_dict(torch.load(model_url, map_location=device))
        print("Model loaded successfully.")
    else:
        print("Pre-trained model not found. Training from scratch.")
    
    return model