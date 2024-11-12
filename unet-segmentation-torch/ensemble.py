import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
unet = model.load_state_dict(torch.load('best_model_benign.pth', map_location=device)))
deeplabv3 = torch.load('best_model_malignant.pth', map_location=device)

unet_model = unet.to(device)
deeplabv3_model = deeplabv3.to(device)

def ensemble_predictions(unet_model, deeplabv3_model, dataloader):
    unet_model.eval()
    deeplabv3_model.eval()
    
    all_ensemble_preds = []
    all_masks = []
    
    with torch.no_grad():
        for images, masks, _ in tqdm(dataloader):
            images = images.to(device)
            
            # Get predictions from both models
            unet_preds = F.softmax(unet_model(images), dim=1)  # Apply softmax for probability distribution
            deeplabv3_preds = F.softmax(deeplabv3_model(images)['out'], dim=1)
            
            # Ensemble by averaging the predictions
            ensemble_preds = (unet_preds + deeplabv3_preds) / 2
            
            # Get the final class predictions by taking the argmax
            final_preds = torch.argmax(ensemble_preds, dim=1).cpu().numpy()
            
            # Collect ensemble predictions and true masks for evaluation
            all_ensemble_preds.extend(final_preds)
            all_masks.extend(masks.cpu().numpy())
    
    return all_ensemble_preds, all_masks

ensemble_predictions(unet_model, deeplabv3_model, DataLoader)