# # main.py
# from pipeline.image_classification_pipeline import landfill_classification_pipeline

# if __name__ == "__main__":
#     pipeline_instance = landfill_classification_pipeline()
#     pipeline_instance.run()

from steps import *
from pipeline.training_pipeline import image_classification_pipeline
from zenml.pipelines import pipeline
import torch
import torch.nn as nn
from torchvision import  models
from steps import *

# Configure your steps here

if __name__ == "__main__":
    pipeline = image_classification_pipeline()
    pipeline.run()