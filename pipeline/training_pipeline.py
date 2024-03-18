from zenml.pipelines import pipeline
import torch
import torch.nn as nn
from torchvision import  models
from steps.data_ingesting import load_data
from steps import *
@pipeline
def training_pipeline():
    transform = define_transformation()
    load_data_step = load_data(transformation=transform,data_dir='imagery')
    initialize_model_step = initialize_model(model_url='C:/Users/Yassine/Documents/Spring 2024/LandfillDetection/LandfillDetection-mlops/models/illegal_landfills_model.pth')
    train_model_step = train_model_with_mlflow(loaders=load_data_step,num_epochs=1, learning_rate=0.001, step_size=7, gamma=0.1,model=initialize_model_step)
    evaluate_model_step = evaluate_model(loaders=load_data_step,model_data=train_model_step)
    model_promoter_step = model_promoter(metrics=evaluate_model_step,precision_threshold=0.8, recall_threshold=0.8)
    log_and_save_step = log_and_save(metrics=evaluate_model_step,model=train_model_step,promotion_decision=model_promoter_step,model_path='C:/Users/Yassine/Documents/Spring 2024/LandfillDetection/LandfillDetection-mlops/models',dvc_file_path='C:/Users/Yassine/Documents/Spring 2024/LandfillDetection/LandfillDetection-mlops/imagery.dvc',registered_model_name='Resnet50')


