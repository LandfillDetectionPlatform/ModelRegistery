from zenml.pipelines import pipeline
import torch
import torch.nn as nn
from torchvision import  models
from steps.data_ingesting import load_data
from steps import *
from PIL import Image
@pipeline
def inference_pipeline():
    image_path = r"C:\Users\Yassine\Documents\Spring 2024\LandfillDetection\LandfillDetection-mlops\imagery\0\image_$32.9942824794879_$-7.68462333146423.png"
    transform = define_transformation()
    initialize_model_step = initialize_model(model_url='C:/Users/Yassine/Documents/Spring 2024/LandfillDetection/LandfillDetection-mlops/models/illegal_landfills_model.pth')
    prediction = run_inference(model=initialize_model_step,image_str=image_path,transformation=transform)
    save_image_based_on_prediction(image_str=image_path,prediction=prediction,save_dir='C:/Users/Yassine/Documents/Spring 2024/LandfillDetection/LandfillDetection-mlops/imagery')

