import os
import glob
from IPython.display import Image, display

# Set inference parameters
weights_path = "runs/train/baxter_yolov5_results/weights/best.pt"  # Path to trained weights
source = "test/images"  # Folder with test images
img_size = 1280
conf_threshold = 0.7

# Run detection
detect_cmd = (
    f"python detect.py --weights {weights_path} --img {img_size} "
    f"--conf {conf_threshold} --source {source}"
)

print("Running inference on test images...")
os.system(detect_cmd)

# Display results
result_images_path = "runs/detect/exp/"
result_images = glob.glob(f"{result_images_path}/*.jpg")
print("\nDisplaying inference results:")
for img_path in result_images:
    display(Image(filename=img_path))
