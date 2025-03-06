import torch
import torch.nn as nn
from torchmetrics.functional import structural_similarity_index_measure as ssim

class SSIMLoss(nn. Module):
    def forward(selfself, img1, img2):
        return 1- ssim(img1, img2)
