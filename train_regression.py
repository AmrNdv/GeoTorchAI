from geotorchai.datasets.raster import SAT4
import os
import numpy as np
import time
import torch
import torch.nn as nn
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms
from geotorchai.models.raster.deepsat2_reg import DeepSatV2_reg
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, SequentialSampler


def createModelAndTrain():
    fullData = TRAIN_DATA

    full_loader = DataLoader(fullData, batch_size= batch_size)
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for i, sample in enumerate(full_loader):
        data_temp, _ = sample
        data_temp = torch.tensor(data_temp, dtype=torch.float)
        channels_sum += torch.mean(data_temp, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(data_temp**2, dim=[0, 2, 3])
        num_batches += 1

    print("Finished loading batches")

    mean = channels_sum / num_batches
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5

    norm_params = {"mean": mean, "std": std}
    np.save(model_dir + '/norm_params.npy', norm_params)
    print("Norm params saved")

    sat_transform = transforms.Normalize(mean, std)

    dataset_size = len(fullData)
    indices = list(range(dataset_size))

    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    training_generator = DataLoader(fullData, **params, sampler=train_sampler)
    val_generator = DataLoader(fullData, **params, sampler=valid_sampler)

    print("Data generators created")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    total_time = 0
    epoch_runnned = 0

    model = DeepSatV2_reg(3, 28, 28)
    if PRE_TRAIN:
        print("model is using pretrained weights as baseline")
        model.load_state_dict(torch.load("models/deepsatv2_regression_tf/model.base.pth"))
    print("Model has been defined")


    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.to(device)
    loss_fn.to(device)

    min_val_metric = float('inf')  # Track lowest validation loss

    # Track losses
    training_losses = []
    validation_losses = []

    for e in range(epoch_nums):
        t_start = time.time()

        # Track training loss for this epoch
        epoch_training_loss = 0
        model.train()

        for i, sample in enumerate(training_generator):
            inputs, labels = sample
            inputs = torch.tensor(inputs, dtype=torch.float).to(device)
            labels = torch.tensor(labels, dtype=torch.float).to(device)

            # Forward pass
            outputs = model(inputs, None)
            loss = loss_fn(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate training loss
            epoch_training_loss += loss.item()
            # for j in range(batch_size):
            #   print("Label: ", labels[j].item(), " | Output: ", outputs[j].item(), " | Difference: ", outputs[j].item()-labels[j].item())


        # Average training loss for this epoch
        epoch_training_loss /= len(training_generator)
        training_losses.append(epoch_training_loss)

        print('Epoch [{}/{}], Training Loss (per sample): {:.10f}'.format(e + 1, epoch_nums, epoch_training_loss))

        # Validation loss
        epoch_validation_loss = 0
        model.eval()
        with torch.no_grad():
            for i, sample in enumerate(val_generator):

                inputs, labels = sample

                inputs = torch.tensor(inputs, dtype=torch.float)
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs, None)
                loss = loss_fn(outputs, labels)
                epoch_validation_loss += loss.item()

        epoch_validation_loss /= len(val_generator)
        validation_losses.append(epoch_validation_loss)

        t_end = time.time()
        total_time += t_end - t_start
        epoch_runnned += 1


        # Validation
        print("Validation Loss (per sample): ", epoch_validation_loss)

        if epoch_validation_loss < min_val_metric:
            min_val_metric = epoch_validation_loss
            torch.save(model.state_dict(), initial_checkpoint)
            print('Best model saved!')

    plot_loss(training_losses, validation_losses)
    return model

def test_regression_model(trained_model):
    TEST_DATA = SAT4(root="data/sat4_regression", is_train_data=False,
                     include_additional_features=False, bands=["red", "green", "blue"],
                     mode="regression")

    # Use a SequentialSampler to ensure order is maintained
    test_sampler = SequentialSampler(TEST_DATA)
    test_generator = DataLoader(TEST_DATA, sampler=test_sampler, **params)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Move the untrained model to the appropriate device
    trained_model.to(device)

    # Initialize variables for regression metrics and plotting
    total_loss_untrained = 0
    all_labels = []
    all_predictions_untrained = []

    # Evaluate the untrained model
    with torch.no_grad():
        for i, sample in enumerate(test_generator):
            inputs, labels = sample

            # Move data to the appropriate device
            inputs = torch.tensor(inputs, dtype=torch.float).to(device)
            labels = torch.tensor(labels, dtype=torch.float).to(device)

            # Forward pass with the untrained model
            outputs = trained_model(inputs, None)

            # Compute loss
            loss_fn = nn.MSELoss()
            loss = loss_fn(outputs, labels)
            total_loss_untrained += loss.item()

            # Collect labels and predictions for plotting
            all_labels.extend(labels.cpu().numpy())
            all_predictions_untrained.extend(outputs.cpu().numpy())

    # Compute loss for the untrained model
    loss_untrained = total_loss_untrained / len(test_generator)

    # Output results
    print("\n************************")
    print(f"Test - Mean Squared  Error (MSE): {loss_untrained:.4f}")

def plot_loss(training_losses, validation_losses):
    # Plot training and validation losses
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epoch_nums + 1), training_losses, label="Training Loss")
    plt.plot(range(1, epoch_nums + 1), validation_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(model_dir + "/loss_plot_MSE.png")  # Save the plot
    plt.show()
    print("Training and validation loss plot saved as 'loss_plot.png'.")

if __name__ == '__main__':
    PRE_TRAIN= True
    TRAIN_DATA = SAT4(root="data/sat4_regression", include_additional_features=False, bands=["red", "green", "blue"],
                      mode="regression")

    epoch_nums = 10  # 350
    learning_rate = 0.0002
    batch_size = 16
    params = {'batch_size': batch_size, 'shuffle': False}

    validation_split = 0.2
    shuffle_dataset = True

    checkpoint_dir = 'models/'
    model_name = 'deepsatv2_regression_tf'
    model_dir = checkpoint_dir + model_name
    os.makedirs(model_dir, exist_ok=True)

    initial_checkpoint = model_dir + '/model.best_MSE.pth'
    LOAD_INITIAL = False
    random_seed = int(time.time())

    model = createModelAndTrain()
    test_regression_model(model)
