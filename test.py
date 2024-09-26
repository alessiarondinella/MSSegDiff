from dataset.ISBI2015 import get_loader
from model import MSSegDiff
import torch
import torch.nn as nn
import os, sys
import argparse
from pathlib import Path
from monai.inferers import SlidingWindowInferer
from trainer import training
import time
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
    parser.add_argument('--feature', type=list, default=[64, 64, 128, 256, 512, 64])
    parser.add_argument('--model', type=str, default='MSSegDiff', help="model backbone: MSSegDiff, MSSegDiff+EncoderSegResNet, MSSegDiff+SegResNet, MSSegDiff+SwinUNETR, MSSegDiff+MultiEncoder")
    
    parser.add_argument('--model_dir', type=str, required=True, help='path in which to find the model')
    parser.add_argument('--model_name', type=str, required=True, help = 'name of the model to test. Example:best_model_*****.pt') #msvalid 6531 isbi 7859
    parser.add_argument('--s', type=int, default=4, help = 'number of outputs to be generated at each step')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--configuration_test', type=str, default='_uncertainty', help= 'inference methods. Possible values: _uncertainty, _mean, _mean-var ')
    parser.add_argument('--resize', type=tuple, default=96)
    parser.add_argument('--testdir', type=str, required=True, help= "directory where to save predictions")

    args = parser.parse_args()
    return args

#SET DEVICE
dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(dev)

if __name__ == '__main__':
    args = parse()
    log = f"FS_{str(args.num_fold)}_{args.configuration}/"
    test_save_path = os.path.join(args.testdir, log)
    os.makedirs(test_save_path, exist_ok=True)

    start_time = time.time()
    #--------------------------------DATALOADER---------------------------------------
    loaders, samplers = get_loader(args)
    print("TEST PATH: ", len(loaders["test"]))

    #tmp = next(iter(loaders["test"]))
    #print(tmp.keys())
    #print(tmp["image"].size()) #torch.Size([1, 3, 156, 197, 152])
    #print(tmp["mask"].size()) #torch.Size([1, 1, 156, 197, 152])

    #--------------------------------TRAINER---------------------------------------
    window_infer = SlidingWindowInferer(roi_size=[96, 96, 96], sw_batch_size=1, overlap=0.6)
    model = MSSegDiff.MSSegDiff_test(args)

    #--------------------------------TEST---------------------------------------
    startEpoch, mean_dsc, history_loss_train, history_loss_val, history_DSC_train, history_DSC_val = training.load_state_dict(model, os.path.join(args.model_dir, args.model_name))
    last_epoch = startEpoch-1
    print("LAST EPOCH: ", last_epoch)
    print("LEN history_DSC_val: ", len(history_DSC_val))
    print("MAX DSC at epoch: ", startEpoch)
    print("BEST DSC: ", mean_dsc)

    
    #sys.exit(0)
    v_mean, _ = training.validation_single_gpu(val_loader=loaders["test"], model=model, device=dev, window_infer=window_infer, save_path=test_save_path, args=args)

    print(f"v_mean is {v_mean}")
    print("--- %s seconds ---" % (time.time() - start_time))