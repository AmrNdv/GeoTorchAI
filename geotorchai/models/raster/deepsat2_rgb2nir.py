import torch
import torch.nn as nn
import torch.nn.functional as F


# class RGB2NIRNet(nn.Module):
#     """
#     An image-to-image translation network that takes a 3-channel (RGB) 28x28 image
#     as input and outputs a 1-channel (NIR) 28x28 image.
#     """
#
#     def __init__(self, in_channels=3, in_height=28, in_width=28):
#         super(RGB2NIRNet, self).__init__()
#
#         # Encoder: similar to the original DeepSatV2 part1.
#         # Note: the ZeroPad2d and MaxPool2d combination keeps the output size equal to the input size.
#         self.encoder = nn.Sequential(
#             nn.Conv2d(in_channels, 32, kernel_size=3, padding="same"),
#             nn.ReLU(),
#             nn.Conv2d(32, 64, kernel_size=3, padding="same"),
#             nn.ReLU(),
#             # nn.ZeroPad2d((in_width // 2, in_width // 2, in_height // 2, in_height // 2)),
#             # nn.MaxPool2d(2),
#             nn.Dropout(0.25)
#         )
#         # After encoder:
#         # For input 28x28, with ZeroPad2d((14,14,14,14)) the feature map becomes 56x56,
#         # then MaxPool2d(2) reduces it back to 28x28.
#         # So encoder output shape: [batch_size, 64, 28, 28].
#
#         # Decoder: convert 64 feature maps back to a 1-channel image.
#         self.decoder = nn.Sequential(
#             nn.Conv2d(64, 32, kernel_size=3, padding="same"),
#             nn.ReLU(),
#             nn.Conv2d(32, 1, kernel_size=3, padding="same")
#             # nn.Sigmoid()
#         )
#
#     def forward(self, images):
#         x = self.encoder(images)
#         x = self.decoder(x)
#         return x

import torch
import torch.nn as nn
import torch.nn.functional as F

class RGB2NIRNet(nn.Module):
    """
    An image-to-image translation network that takes a 3-channel (RGB) 28x28 image
    as input and outputs a 1-channel (NIR) 28x28 image.
    """

    def __init__(self, in_channels=3, in_height=28, in_width=28):
        super(RGB2NIRNet, self).__init__()

        # Encoder: similar to the original DeepSatV2 part1.
        # Note: the ZeroPad2d and MaxPool2d combination keeps the output size equal to the input size.
        self.sequences_part1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding="same"),
            nn.ReLU(),
            # nn.ZeroPad2d((in_width // 2, in_width // 2, in_height // 2, in_height // 2)),
            # nn.MaxPool2d(2),
            nn.Dropout(0.25)
        )
        # After encoder:
        # For input 28x28, with ZeroPad2d((14,14,14,14)) the feature map becomes 56x56,
        # then MaxPool2d(2) reduces it back to 28x28.
        # So encoder output shape: [batch_size, 64, 28, 28].

        # Decoder: convert 64 feature maps back to a 1-channel image.
        self.sequences_part2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=3, padding="same")
            # nn.Sigmoid()
        )

    def forward(self, images):
        x = self.sequences_part1(images)
        x = self.sequences_part2(x)
        return x