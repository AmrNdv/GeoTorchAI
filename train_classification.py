from geotorchai.datasets.raster import SAT4
import torch
from torch.utils.data import Subset
import os
import numpy as np
import time
import torch
import torch.nn as nn
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from geotorchai.models.raster import DeepSatV2
import matplotlib.pyplot as plt




def createModelAndTrain():
    fullData = TRAIN_DATA_CLASSIFICATION # need to change to FULL_DATA
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
    # fullData = SAT4(root = "data/sat4", include_additional_features = True, transform = sat_transform)
    # print("Dataset had been loaded")

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

    model = DeepSatV2(3, 28, 28, 4)
    print("Model has been defined")

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.to(device)
    loss_fn.to(device)

    max_val_accuracy = None
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
            #print(inputs[0].shape)

            inputs = torch.tensor(inputs, dtype=torch.float)

            inputs = inputs.to(device)
            #features = features.type(torch.FloatTensor).to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(inputs, None)
            loss = loss_fn(outputs, labels)

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
        print('Epoch [{}/{}], Training Loss: {:.4f}'.format(e + 1, epoch_nums, loss.item()))

        val_accuracy = get_validation_accuracy(model, val_generator, device)
        print("Validation Accuracy: ", val_accuracy, "%")

        if max_val_accuracy == None or val_accuracy > max_val_accuracy:
            max_val_accuracy = val_accuracy
            torch.save(model.state_dict(), initial_checkpoint)
            print('best model saved!')

    plot_loss(training_losses, validation_losses)
    return model

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
    plt.savefig(model_dir + "/loss_plot.png")  # Save the plot
    plt.show()
    print("Training and validation loss plot saved as 'loss_plot.png'.")

def get_validation_accuracy(model, val_generator, device):
    model.eval()
    total_sample = 0
    correct = 0
    for i, sample in enumerate(val_generator):
        inputs, labels = sample

        inputs = torch.tensor(inputs, dtype=torch.float)

        inputs = inputs.to(device)
        #features = features.type(torch.FloatTensor).to(device)
        labels = labels.to(device)

        outputs = model(inputs, None)
        total_sample += len(labels)

        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()

    accuracy = 100 * correct / total_sample

    return accuracy

def test_classification_model(trained_model):
    TEST_DATA = SAT4(root="data/sat4_classification", is_train_data=False, include_additional_features=False,
                     bands=["red", "green", "blue"])
    subset_size = int(len(TEST_DATA))  # 10% of your dataset
    test_indices = torch.randperm(subset_size)[:subset_size]  # Random indices for subset
    test_subset = Subset(TEST_DATA, test_indices)  # Creating the subset
    test_sampler = SubsetRandomSampler(test_indices)
    test_generator = DataLoader(test_subset, **params, sampler=test_sampler)

    total_sample = 0
    correct = 0

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    trained_model.to(device)

    for i, sample in enumerate(test_generator):
        inputs, labels = sample

        inputs = torch.tensor(inputs, dtype=torch.float)

        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = trained_model(inputs, None)
        total_sample += len(labels)

        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()

    accuracy = 100 * correct / total_sample
    print(f"Test Accuracy for classification is: {accuracy}")


if __name__ == '__main__':
    TRAIN_DATA_CLASSIFICATION = SAT4(root="data/sat4_classification", include_additional_features=False,
                                     bands=["red", "green", "blue"])

    epoch_nums = 10
    learning_rate = 0.0002
    batch_size = 16
    params = {'batch_size': batch_size, 'shuffle': False}

    validation_split = 0.2
    shuffle_dataset = True

    checkpoint_dir = 'models/'
    model_name = 'deepsatv2_classification'
    model_dir = checkpoint_dir + "/" + model_name
    os.makedirs(model_dir, exist_ok=True)

    initial_checkpoint = model_dir + '/model.best.pth'
    LOAD_INITIAL = False
    random_seed = int(time.time())

    model = createModelAndTrain()
    test_classification_model(model)


