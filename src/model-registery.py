import torch
import mlflow
import mlflow.pytorch
from torchvision import models, transforms

# Set the tracking URI
mlflow.set_tracking_uri("http://localhost:5000")  # replace with your tracking URI

# Load the pre-trained ResNet50 model
model_path = "../models/illegal_landfills_model.pth"
model = models.resnet50(weights=None)  # Replace deprecated 'pretrained' with 'weights=None'
num_classes = 2  # Legal and Illegal landfills
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

# Load the state dictionary
state_dict = torch.load(model_path, map_location=torch.device('cpu'))

# Apply the state dictionary to the model
model.load_state_dict(state_dict)

# Define example input tensor for schema
example_input = torch.randn(1, 3, 224, 224)
# Convert PyTorch tensor to NumPy array
example_input_numpy = example_input.numpy()

# Start an MLflow run
with mlflow.start_run() as run:
    # Log model without automatic signature inference
    mlflow.pytorch.log_model(model, "model", registered_model_name="Resnet50", input_example=example_input_numpy, signature=False)

    # Register the model in the model registry
    mlflow.register_model(f"runs:/{run.info.run_id}/model", "ResNet50")

    # Log additional information
    mlflow.log_param("description", "ResNet-50 model for image classification of landfills")
    mlflow.log_param("epochs", 200)
    mlflow.log_param("frozen_layers", "All layers frozen except the last one")
    mlflow.log_param("dataset", "Aerial waste dataset")

    # Log metrics
    precision = 0.96
    recall = 0.97
    accuracy = 0.94

    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("accuracy", accuracy)
