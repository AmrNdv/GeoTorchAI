import os
import numpy as np
import pandas as pd
import time
import torch
import torch.nn as nn
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader, SequentialSampler
from geotorchai.models.raster.deepsat2_rgb2nir import RGB2NIRNet
import matplotlib.pyplot as plt
from geotorchai.datasets.raster import SAT4



def createModelAndTrain():
    fullData = TRAIN_DATA_CONSTRUCTION
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
        print("model is working with cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("model is working with mps")
    else:
        device = torch.device("cpu")
        print("model is working with cpu")


    total_time = 0
    epoch_runnned = 0

    model = RGB2NIRNet(3, 28, 28)
    if PRE_TRAIN:
        model.load_state_dict(torch.load("models/deepsatv2_construction_tf/model.base.pth"))

    print("Model has been defined")

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.to(device)
    loss_fn.to(device)

    min_val_metric = float('inf')
    # Track losses
    training_losses = []
    validation_losses = []

    for e in range(epoch_nums):
        t_start = time.time()

        # Track training loss for this epoch
        epoch_training_loss = 0
        model.train()

        for i, sample in enumerate(training_generator):
            inputs, targets = sample
            #print(inputs[0].shape)

            inputs = torch.tensor(inputs, dtype=torch.float).to(device)
            targets = torch.tensor(targets, dtype=torch.float).to(device)
            targets = targets.view(-1,1,28,28)

            # Forward pass
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate training loss
            epoch_training_loss += loss.item()

        # Average training loss for this epoch
        epoch_training_loss /= len(training_generator)
        training_losses.append(epoch_training_loss)

        # Validation loss
        epoch_validation_loss = 0
        model.eval()
        with torch.no_grad():
            for i, sample in enumerate(val_generator):
                inputs, targets = sample

                inputs = torch.tensor(inputs, dtype=torch.float).to(device)
                targets = torch.tensor(targets, dtype=torch.float).to(device)
                targets = targets.view(-1, 1, 28, 28)

                outputs = model(inputs)
                loss = loss_fn(outputs, targets)
                epoch_validation_loss += loss.item()

        epoch_validation_loss /= len(val_generator)
        validation_losses.append(epoch_validation_loss)

        t_end = time.time()
        total_time += t_end - t_start
        epoch_runnned += 1
        print('Epoch [{}/{}], Training Loss: {:.4f}'.format(e + 1, epoch_nums, epoch_training_loss))

        val_metric = get_validation_metric(model, val_generator, device)
        print("Validation Metric: ", val_metric)

        if val_metric < min_val_metric:
            min_val_metric = val_metric
            torch.save(model.state_dict(), initial_checkpoint)
            print(f'val metric is:{val_metric} - model saved')

    # Plot training and validation losses
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epoch_nums + 1), training_losses, label="Training Loss")
    plt.plot(range(1, epoch_nums + 1), validation_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(model_dir + "/loss_plot.png")  # Save the plot
    plt.show()
    print("Training and validation loss plot saved as 'loss_plot.png'.")

def get_validation_metric(model, val_generator, device):
    model.eval()
    eval_loss= nn.MSELoss()
    total_loss = 0
    for i, sample in enumerate(val_generator):
        inputs, targets = sample

        inputs = torch.tensor(inputs, dtype=torch.float).to(device)
        targets = torch.tensor(targets, dtype=torch.float).to(device)
        targets = targets.view(-1, 1, 28, 28)

        outputs = model(inputs)
        loss = eval_loss(outputs, targets)
        total_loss += loss.item()

    total_loss /= len(val_generator)
    return total_loss

def test_construction_model(trained_model):
    TEST_DATA = SAT4(root="data/sat4_construction", is_train_data=False,
                     include_additional_features=False, bands=["red", "green", "blue"],
                     mode="construction")

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
            outputs = trained_model(inputs)

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

if __name__ == '__main__':
    batch_size = 16
    params = {'batch_size': batch_size, 'shuffle': False}
    model = RGB2NIRNet(3, 28, 28)
    model.load_state_dict(torch.load("models/deepsatv2_construction_without_padding_tf/model.best_20_epochs.pth"))
    test_construction_model(model)



    # PRE_TRAIN=True
    # # Load Training data:
    # print("Start loading data!")
    # start = time.time()
    # TRAIN_DATA_CONSTRUCTION = SAT4(root="data/sat4_construction", include_additional_features=False,
    #                                  bands=["red", "green", "blue"],mode='construction')
    # finish = time.time()
    # print("Finish loading data!")
    # print(f"Loading data took {(finish-start)/1000} seconds")
    #
    #
    # epoch_nums = 20  # 350
    # learning_rate = 0.0002
    # batch_size = 16
    # params = {'batch_size': batch_size, 'shuffle': False}
    #
    # validation_split = 0.2
    # shuffle_dataset = True
    #
    # checkpoint_dir = 'models/'
    # model_name = 'deepsatv2_construction_tf'
    # model_dir = checkpoint_dir + "/" + model_name
    # os.makedirs(model_dir, exist_ok=True)
    # random_seed = int(time.time())
    # print("Initiating create model and train!")
    # initial_checkpoint = model_dir + '/model.best_20_epochs.pth'
    # createModelAndTrain()


