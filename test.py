from setting import parse_opts 
from datasets.brains18 import BrainS18Dataset
from model import generate_model
import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F
from scipy import ndimage
import nibabel as nib
import sys
import os
from utils.path_security import safe_image_path
from utils.checkpoint import load_checkpoint
import numpy as np


def seg_eval(pred, label, clss):
    """
    calculate the dice between prediction and ground truth
    input:
        pred: predicted mask
        label: groud truth
        clss: eg. [0, 1] for binary class
    """
    Ncls = len(clss)
    dices = np.zeros(Ncls)
    [depth, height, width] = pred.shape
    for idx, cls in enumerate(clss):
        # binary map
        pred_cls = np.zeros([depth, height, width])
        pred_cls[np.where(pred == cls)] = 1
        label_cls = np.zeros([depth, height, width])
        label_cls[np.where(label == cls)] = 1

        # cal the inter & conv
        s = pred_cls + label_cls
        inter = len(np.where(s >= 2)[0])
        conv = len(np.where(s >= 1)[0]) + inter
        try:
            dice = 2.0 * inter / conv
        except:
            print("conv is zeros when dice = 2.0 * inter / conv")
            dice = -1

        dices[idx] = dice

    return dices

def test(data_loader, model, img_names, sets):
    masks = []
    model.eval() # for testing 
    for batch_id, batch_data in enumerate(data_loader):
        # forward
        volume = batch_data
        if not sets.no_cuda:
            volume = volume.cuda()
        with torch.no_grad():
            probs = model(volume)
            probs = F.softmax(probs, dim=1)

        # resize mask to original size
        [batchsize, _, mask_d, mask_h, mask_w] = probs.shape
        data = nib.load(safe_image_path(sets.data_root, img_names[batch_id]))
        data = np.asanyarray(data.dataobj)
        [depth, height, width] = data.shape
        mask = probs[0].detach().cpu().numpy()
        scale = [1, depth*1.0/mask_d, height*1.0/mask_h, width*1.0/mask_w]
        mask = ndimage.zoom(mask, scale, order=1)
        mask = np.argmax(mask, axis=0)
        
        masks.append(mask)
 
    return masks


if __name__ == '__main__':
    # settting
    sets = parse_opts()
    sets.target_type = "normal"
    sets.phase = 'test'

    # getting model
    checkpoint = load_checkpoint(sets.resume_path)
    net, _ = generate_model(sets)
    net.load_state_dict(checkpoint['state_dict'])

    # data tensor
    testing_data =BrainS18Dataset(sets.data_root, sets.img_list, sets)
    data_loader = DataLoader(testing_data, batch_size=1, shuffle=False, num_workers=1, pin_memory=False)

    # testing
    rows = [info.split() for info in testing_data.img_list]
    if any(len(row) != 2 for row in rows):
        raise ValueError("Evaluation rows must contain an image and a label.")
    img_names = [row[0] for row in rows]
    masks = test(data_loader, net, img_names, sets)
    
    # evaluation: calculate dice 
    label_names = [row[1] for row in rows]
    Nimg = len(label_names)
    dices = np.zeros([Nimg, sets.n_seg_classes])
    for idx in range(Nimg):
        label = nib.load(safe_image_path(sets.data_root, label_names[idx]))
        label = np.asanyarray(label.dataobj)
        dices[idx, :] = seg_eval(masks[idx], label, range(sets.n_seg_classes))
    
    # print result
    for idx in range(1, sets.n_seg_classes):
        mean_dice_per_task = np.mean(dices[:, idx])
        print('mean dice for class-{} is {}'.format(idx, mean_dice_per_task))   
