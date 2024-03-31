import argparse
from pipeline.training_pipeline import training_pipeline
from pipeline.inference_pipeline import inference_pipeline
from PIL import Image
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Choose pipeline to run")
    parser.add_argument('--pipeline', choices=['training', 'inference'], default='training',
                        help="Choose the pipeline to run (training or inference)")
    args = parser.parse_args()

    if args.pipeline == 'training':
        pipeline = training_pipeline()
    elif args.pipeline == 'inference':
        pipeline = inference_pipeline().with_options(image_path="C:\\Users\\Yassine\\Documents\\Spring 2024\\LandfillDetection\\LandfillDetection-mlops\\imagery\\0\\image_$32.9942824794879_$-7.68462333146423.png")
            #image_path = r""

    else:
        raise ValueError("InValid pipeline choice. Choose 'training' or 'inference'.")

    pipeline.run()
