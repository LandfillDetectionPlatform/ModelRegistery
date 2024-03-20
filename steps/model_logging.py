import mlflow
import os
from zenml import step
import mlflow.pytorch
import torch
from mlflow.tracking import MlflowClient
import yaml  
@step(enable_cache=False)
def log_and_save(model: dict, metrics: dict, promotion_decision: bool, model_path: str, dvc_file_path: str, registered_model_name: str):

    mlflow.set_tracking_uri("http://localhost:5000")
    precision = metrics["precision"]
    recall = metrics["recall"]
    accuracy = metrics["accuracy"]
    f1 = metrics["f1_score"]

    model_container = model["model"]
    hyperparameters = model["hyperparameters"]
    
    client = MlflowClient()
    
    # Logic to increment model version
    latest_version_info = client.get_latest_versions(registered_model_name, stages=["None"])
    if latest_version_info:
        latest_version = max([int(version.version) for version in latest_version_info])
        new_version = latest_version + 1
    else:
        new_version = 1
    
    # Path where the model will be stored
    model_artifact_path = os.path.join(model_path, f"model_v{new_version}.pth")
    
    if promotion_decision:
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(model_container.state_dict(), model_artifact_path)
        mlflow.end_run()
        # Start an MLflow run
        with mlflow.start_run() as run:
                # Log metrics
            mlflow.log_metric("precision", 0.4)
            mlflow.log_metric("recall", 0.6)
            mlflow.log_metric("accuracy", 0.34)
            mlflow.log_metric("f1_score", 0.5)
            mlflow.log_param("num_epochs",hyperparameters['num_epochs'])
            mlflow.log_param("learning_rate",hyperparameters['learning_rate'])
            mlflow.log_param("step_size",hyperparameters['step_size'])
            mlflow.log_param("gamma",hyperparameters['gamma'])
            # # Log model parameters and hyperparameters
            # for param, value in model_params.items():
            #     mlflow.log_param(param, value)


            # Log the model to MLflow
            mlflow.pytorch.log_model(model_container, "model", registered_model_name=registered_model_name)
            
            # Optionally, register the model in the model registry
            mlflow.register_model(f"runs:/{run.info.run_id}/model", registered_model_name)
            
            # If a .dvc file exists, log it as an artifact
            if os.path.exists(dvc_file_path):
                mlflow.log_artifact(dvc_file_path, "dvc_files")
            dataset_version = "unknown"
            if os.path.exists(dvc_file_path):
                mlflow.log_artifact(dvc_file_path, "dvc_files")
                try:
                    with open(dvc_file_path, 'r') as file:
                        dvc_content = yaml.safe_load(file)  # Use yaml.safe_load to parse the .dvc file
                        # Extract dataset version or hash; adapt this based on your .dvc file structure
                        dataset_version = dvc_content.get('outs', [{}])[0].get('md5', 'unknown')
                except Exception as e:
                    print(f"Error processing .dvc file {dvc_file_path}: {e}")

            
            mlflow.log_param("dataset_version", dataset_version)  

            print(f"Model v{new_version} saved and registered under '{registered_model_name}' with MLflow.")

    else:
        print("Model not promoted.")
