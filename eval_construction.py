import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.distributed.checkpoint  # if needed
from torch.utils.data import DataLoader, Subset

from geotorchai.datasets.raster import SAT4
from geotorchai.models.raster.deepsat2_rgb2nir import RGB2NIRNet


def eval_construction(model, val_generator):
    model.eval()
    return_list = []
    for i, sample in enumerate(val_generator):
        inputs, labels = sample
        inputs = torch.tensor(inputs, dtype=torch.float)
        outputs = model(inputs)
        return_list.append(outputs)
    return return_list


def load_csv_data(num_images, x_path, y_path):
    X = pd.read_csv(x_path, header=None,skiprows=SKIP, nrows=num_images).values.reshape(
        [num_images, 28, 28, 4]
    )
    Y = pd.read_csv(y_path, header=None,skiprows=SKIP, nrows=num_images).values.reshape(
        [num_images, 28, 28, 1]
    )
    return X, Y


def load_model(model_path):
    model = RGB2NIRNet(3, 28, 28)
    model.load_state_dict(torch.load(model_path))
    return model


def get_val_generator(num_images, root="data/sat4_construction"):
    # Create SAT4 test dataset with specified bands and configuration.
    testData = SAT4(
        root=root,
        include_additional_features=False,
        bands=["red", "green", "blue"],
        is_train_data=False
    )
    dataset_size = len(testData)
    N = min(num_images, dataset_size)
    subset_data = Subset(testData, list(range(SKIP, SKIP + N)))
    params = {'batch_size': num_images, 'shuffle': False}
    return DataLoader(subset_data, **params)


def get_predictions(model, val_generator):
    # Get predictions from the model using the validation generator.
    outputs = eval_construction(model, val_generator)[0]
    # Permute dimensions and convert to numpy for plotting.
    predictions = outputs.permute(0, 2, 3, 1).detach().cpu().numpy()
    return predictions


def plot_results(X, Y, predictions, num_images):
    # Create subplot grid: one row per image and three columns
    fig, axes = plt.subplots(nrows=num_images, ncols=3, figsize=(10, num_images * 3))

    # Add column titles on the first row
    axes[0, 0].set_title("RGB image", fontsize=20)
    axes[0, 1].set_title("NIR image", fontsize=20)
    axes[0, 2].set_title("NIR prediction based on RGB image", fontsize=20)

    for i in range(num_images):
        # Display input image (only RGB channels)
        ax_img = axes[i, 0]
        ax_img.imshow(X[i, :, :, :3])
        ax_img.axis('off')

        # Display ground truth (NIR image in grayscale)
        ax_gt = axes[i, 1]
        ax_gt.imshow(Y[i, :, :, 0], cmap="inferno")
        ax_gt.axis('off')

        # Display prediction (NIR prediction in grayscale)
        ax_pred = axes[i, 2]
        ax_pred.imshow(predictions[i, :, :, 0], cmap="inferno")
        ax_pred.axis('off')

    plt.tight_layout()
    plt.show()



def main():
    NUM_IMAGES = 3
    # Load CSV data for inputs (X) and labels (Y)
    X, Y = load_csv_data(NUM_IMAGES, "data/sat4_construction/x_test_sat4.csv",
                         "data/sat4_construction/y_test_sat4.csv")

    # Load the pretrained model
    model = load_model("models/deepsatv2_construction_without_padding/model.best_20_epochs.pth")

    # Create the validation data loader
    val_generator = get_val_generator(NUM_IMAGES)

    # Generate predictions using the model
    predictions = get_predictions(model, val_generator)

    # Plot the original images, true labels, and predicted labels
    plot_results(X, Y, predictions, NUM_IMAGES)
if __name__ == '__main__':
    SKIP = 1947+15
    print(SKIP)
    main()
