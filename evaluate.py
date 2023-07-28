import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm

from utils.dice_score import multiclass_dice_coeff, dice_coeff
from utils.dice_score import dice_loss

@torch.inference_mode()
def evaluate(net, dataloader, device, amp, n_classes):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0
    epoch_loss = 0
    # iterate over the validation set
    criterion = nn.CrossEntropyLoss() if n_classes > 1 else  nn.BCEWithLogitsLoss()
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true = batch['image'], batch['mask']
            #mask_true = mask_true.squeeze(1)

            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32) #, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)

            # predict the mask
            mask_pred = net(image)

            if n_classes == 1:
                assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'
                # loss = criterion(mask_pred.squeeze(1), mask_true.squeeze(1).float())
                # loss += dice_loss(F.sigmoid(mask_pred.squeeze(1)), mask_true.squeeze(1).float(), multiclass=False)                
                # epoch_loss += loss.item()
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                # compute the Dice score
                dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
            else:
                assert mask_true.min() >= 0 and mask_true.max() < n_classes, 'True mask indices should be in [0, n_classes['
                # loss = criterion(mask_pred.squeeze(1), mask_true.float())
                # loss += dice_loss(F.sigmoid(mask_pred.squeeze(1)), mask_true.float(), multiclass=False)
                # epoch_loss += loss.item()                
                # convert to one-hot format
                mask_true = F.one_hot(mask_true, n_classes).permute(0, 3, 1, 2).float()
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                dice_score += multiclass_dice_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)

    net.train()
    return dice_score / max(num_val_batches, 1)#, epoch_loss
