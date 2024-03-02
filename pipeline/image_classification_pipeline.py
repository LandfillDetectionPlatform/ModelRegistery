from zenml.pipelines import pipeline
from steps.data_ingesting import load_data
from steps.model_training import train_model_with_mlflow

@pipeline
def landfill_classification_pipeline():
    loaders = load_data(data_dir="C:/Users/Yassine/Documents/Spring 2024/LandfillDetection/LandfillDetection-mlops/imagery")
    train_model_with_mlflow(loaders=loaders, num_epochs=10,learning_rate=0.001, model_url="C:/Users/Yassine/Documents/Spring 2024/LandfillDetection/LandfillDetection-mlops/models/illegal_landfills_model.pth", step_size=7, gamma=0.1, output_model_path="C:/Users/Yassine/Documents/Spring 2024/LandfillDetection/LandfillDetection-mlops/models")