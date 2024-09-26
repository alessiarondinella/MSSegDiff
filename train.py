import os, sys
import argparse
from pathlib import Path
import torch
import torch.nn as nn
from dataset.ISBI2015 import get_loader
from model import MSSegDiff
from monai.inferers import SlidingWindowInferer
from trainer.utils.lr_scheduler import LinearWarmupCosineAnnealingLR
from monai.losses.dice import DiceLoss, DiceFocalLoss
from trainer import training
from trainer.utils.modules import datiCliniciLateFusion
import matplotlib.pyplot as plt

def parse():
    '''Returns args passed to the train.py script.'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=Path, required=True, help="Path to the ISBI2015 dataset")
    parser.add_argument('--folds_path', type=Path, default='dataset/2-dataset_crossvalidation_5fold.json', help='path to the json file with fold splitting')
    parser.add_argument('--configuration', type=str, default='SA')
    parser.add_argument('--num_fold', type=int, default=0)
    parser.add_argument('--number_modality', type=int, default=3)
    parser.add_argument('--number_targets', type=int, default=1)
    parser.add_argument('--feature', type=list, default=[64, 64, 128, 256, 512, 64], help= "for BasicUnet [64, 64, 128, 256, 512, 64], for SegResNet [64, 128, 256, 512, 1024, 64]")
    parser.add_argument('--model_name', type=str, default='')

    parser.add_argument('--resize', type=tuple, default=96)

    parser.add_argument('--model', type=str, default='MSSegDiff', help="model backbones: MSSegDiff, MSSegDiff+EncoderSegResNet, MSSegDiff+SegResNet, MSSegDiff+SwinUNETR, MSSegDiff+MultiEncoder")
    parser.add_argument('--learning_rate', type=int, default=1e-4)#1e-4

    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--max_epoch', type=int, default=500)
    parser.add_argument('--logdir', type=str, required=True, help = 'Directory were to save log and models')
    parser.add_argument('--val_every', type=int, default=10)
    parser.add_argument('--resume')
    parser.add_argument('--experiment', type=str, default='1')

    args = parser.parse_args()
    return args

#SET DEVICE
dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(dev)

if __name__ == '__main__':
    args = parse()
    log = f"FS_{str(args.num_fold)}_{args.configuration}/"
    logdir = os.path.join(args.logdir, log)
    model_save_path = os.path.join(logdir, "model")
    os.makedirs(model_save_path, exist_ok=True)
    
    # -----------------------------Dataset e Loader---------------------------------------
    print("sono qui")
    loaders, samplers = get_loader(args)
    print("TRAIN PATH: ", len(loaders['train']))
    print("VAL PATH: ", len(loaders['val']))
    print("TEST PATH: ", len(loaders['test']))
    
    '''
    #CHECK INPUT
    tmp = next(iter(loaders["train"]))
    print(tmp["images"].size()) #torch.Size([2, 3, 96, 96, 96])
    print(tmp["mask"].size()) #torch.Size([2, 1, 96, 96, 96])
    print(tmp["images_meta_dict"]["filename_or_obj"])
    plt.imsave("T1.png", tmp["images"][0][0][:,:,90])
    plt.imsave("T2.png", tmp["images"][0][1][:,:,90])
    plt.imsave("FLAIR.png", tmp["images"][0][2][:,:,90])
    plt.imsave("MASK.png", tmp["mask"][0][0][:,:,90])
    sys.exit(0)
    '''
    #--------------------------------TRAINER---------------------------------------

    window_infer = SlidingWindowInferer(roi_size=[96, 96, 96], sw_batch_size=1, overlap=0.25)###0.25
    model = MSSegDiff.MSSegDiff(args)
    
    #print(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-3)
    ce = None #nn.CrossEntropyLoss() 
    mse = nn.MSELoss() #None
    scheduler = LinearWarmupCosineAnnealingLR(optimizer,
                                                  warmup_epochs=args.max_epoch/10,
                                                  max_epochs=args.max_epoch)
    bce = nn.BCEWithLogitsLoss()
    dice_loss = DiceLoss(include_background = False, sigmoid=True) ###ho escluso il background!!!!
    #dice_focal_loss = DiceFocalLoss(include_background = False, sigmoid=True)


    #--------------------------------TRAIN NEW---------------------------------------
    best_mean_dice = 0.0
    if args.model_name == "":
        startEpoch=0
        training.train(train_loader=loaders['train'], val_loader=loaders['val'], model = model, optimizer=optimizer, 
                       scheduler = scheduler, model_save_path=model_save_path, start_epoch = startEpoch, device=dev, 
                       max_epochs=args.max_epoch, dice_loss=dice_loss, bce=bce, mse=mse,
                       window_infer=window_infer, best_mean_dice=best_mean_dice, args=args)
    else:
        startEpoch, mean_dice, epoch_loss, epoch_accuracy = training.load_state_dict(model, os.path.join(model_save_path, args.model_name))
        last_epoch = startEpoch-1
        max_epoch = args.max_epoch - last_epoch
        print("LAST EPOCH: ", last_epoch)
        print("mean_dice: ", mean_dice)
        #print("LEN history_DSC_val: ", len(epoch_accuracy["val"]))
        print("history_DSC_val: ", epoch_accuracy)
        print("for epoch in range: ", startEpoch, " - ", startEpoch+args.max_epoch)
        print("MAX EPOCH: ", max_epoch)
        #sys.exit(0)
        training.train(train_loader=loaders['train'], val_loader=loaders['val'], model = model, optimizer=optimizer, 
                       scheduler = scheduler, model_save_path=model_save_path, start_epoch = startEpoch, device=dev, 
                       max_epochs=max_epoch, dice_loss=dice_loss, bce=bce, mse=mse,
                       val_every=args.val_every, window_infer=window_infer, best_mean_dice=best_mean_dice, args=args)
        
    

