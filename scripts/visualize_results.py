import os
import matplotlib.pyplot as plt
from IPython.display import Image, display

# Path to the results.txt file generated during training
results_path = "runs/train/baxter_yolov5_results/results.png"
results_txt_path = "runs/train/baxter_yolov5_results/results.txt"

# Function to plot training results manually (if plot_results() isn't available)
def plot_results_manually(results_txt_path):
    try:
        # Load the results from the text file
        with open(results_txt_path, 'r') as f:
            lines = f.readlines()

        # Extract the lines corresponding to training losses and mAP (modify as necessary)
        epochs = []
        loss = []
        mAP = []

        for line in lines:
            if 'Epoch' in line:  # Look for lines containing epoch info
                parts = line.split()
                epochs.append(int(parts[0]))
                loss.append(float(parts[2]))  # Assuming the loss value is in the third column
                mAP.append(float(parts[5]))  # Assuming mAP@0.5 is in the sixth column

        # Plot training loss and mAP
        fig, ax1 = plt.subplots(figsize=(10, 6))

        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss', color='tab:red')
        ax1.plot(epochs, loss, color='tab:red', label='Loss')
        ax1.tick_params(axis='y', labelcolor='tab:red')

        ax2 = ax1.twinx()  # Instantiate a second y-axis sharing the same x-axis
        ax2.set_ylabel('mAP@0.5', color='tab:blue')  # Make the label of the second axis blue
        ax2.plot(epochs, mAP, color='tab:blue', label='mAP@0.5')
        ax2.tick_params(axis='y', labelcolor='tab:blue')

        # Add title and show the plot
        plt.title('Training Loss and mAP over Epochs')
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Error plotting results: {e}")

# Run the custom plotting function
plot_results_manually(results_txt_path)

# Visualize the final results image
try:
    display(Image(filename=results_path, width=1000))
    print(f"Training metrics plotted. Check {results_path}.")
except FileNotFoundError:
    print("Error: Results file not found. Ensure training has completed.")
