import copy
import torch

from geotorchai.models.raster.deepsat2_rgb2nir import RGB2NIRNet
from geotorchai.models.raster.deepsat2_reg import DeepSatV2_reg


def load_pre_trained_weights(trained_model_path, blank_model, save_path, restrict_seq2=False):
    # Load the pretrained weights from trained model
    pretrained_weights = torch.load(trained_model_path)

    # Load the state_dict of the model
    new_model_state_dict = blank_model.state_dict()
    basic_model_state_dir = copy.deepcopy(new_model_state_dict)
    # Update weights only for aligned layers
    for name, param in pretrained_weights.items():
        if name.startswith("sequences_part2") and restrict_seq2:
            continue  # Skip instances that don't start with "blablabla"

        if name in new_model_state_dict and new_model_state_dict[name].shape == param.shape:
            new_model_state_dict[name] = param  # Copy matching weights
            print(f"Matching layer: {name}")
        else:
            print(f"Skipping layer: {name} (mismatch or not present in model_2)")

    # Load the updated state_dict into the model
    blank_model.load_state_dict(new_model_state_dict)

    torch.save(blank_model.state_dict(), save_path)

if __name__ == '__main__':
    load_pre_trained_weights("models/deepsatv2_classification/model.best.pth",
                             RGB2NIRNet(3, 28, 28),
                             "models/deepsatv2_construction_tf/model.base.pth",
                             restrict_seq2=True)