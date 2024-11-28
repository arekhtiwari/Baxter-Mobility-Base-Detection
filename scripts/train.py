import os

# Set up training parameters
img_size = 1280
batch_size = 8
epochs = 100
data_yaml = "data.yaml"  # Path to dataset YAML file
model_cfg = "./models/custom_yolov5l.yaml"  # Path to custom YOLOv5 config
output_name = "baxter_yolov5_results"  # Experiment name

# Command to run training
train_cmd = (
    f"python train.py --img {img_size} --batch {batch_size} --epochs {epochs} "
    f"--data {data_yaml} --cfg {model_cfg} --weights '' --name {output_name} --cache disk"
)

print("Starting YOLOv5 training...")
os.system(train_cmd)
print("Training completed!")
