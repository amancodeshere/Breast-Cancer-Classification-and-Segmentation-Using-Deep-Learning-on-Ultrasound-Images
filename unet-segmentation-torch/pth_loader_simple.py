# Gavin
# To use this script, you need to have pytorch installed.
# Type python pth_loader_simple.py in the terminal to run the script.

import torch
# model = torch.load('best_model_benign.pth')
model = torch.load('best_model_malignant.pth')
print(model)