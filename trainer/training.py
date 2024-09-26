import numpy as np
import os
import glob
import torch
from tqdm import tqdm
from monai.utils import set_determinism
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from trainer.evaluation.metric import dice, hausdorff_distance_95, recall, fscore, precision, false_positive_rate, ltpr, lfpr, compute_avd, avg_surface_distance_symmetric
from monai.data import DataLoader
from scipy import ndimage as ndi
import scipy.ndimage as ndimage
import nibabel as nib
import time
import subprocess

import torch.nn.functional as F


def load_state_dict(model, weight_path, strict=True):
        sd = torch.load(weight_path, map_location="cpu")
        if "module" in sd :
            sd = sd["module"]
        new_sd = {}
        for k, v in sd.items():
            k = str(k)
            new_k = k[7:] if k.startswith("module") else k 
            new_sd[new_k] = v 

        startEpoch = new_sd['startEpoch']
        mean_dsc = new_sd['mean_dsc']
        history_loss_t = new_sd['history_loss_trn']
        history_loss_v = new_sd['history_loss_vld']
        history_DSC_t = new_sd['history_DSC_trn']
        history_DSC_v = new_sd['history_DSC_vld']

        model.load_state_dict(new_sd['model_state'], strict=strict)
        
        print(f"model parameters are loaded successed.")
        return startEpoch, mean_dsc, history_loss_t, history_loss_v, history_DSC_t, history_DSC_v

def delete_last_model(model_dir, symbol):

    last_model = glob.glob(f"{model_dir}/{symbol}*.pt")
    if len(last_model) != 0:
        os.remove(last_model[0])


def save_new_model_and_delete_last(model, epoch, mean_dice, history_loss_trn, history_loss_vld, history_DSC_trn, history_DSC_vld, save_path, delete_symbol=None):
    save_dir = os.path.dirname(save_path)

    os.makedirs(save_dir, exist_ok=True)
    if delete_last_model is not None:
        delete_last_model(save_dir, delete_symbol)
    
    #torch.save(model.state_dict(), save_path)
    torch.save({
        'startEpoch': epoch + 1,
        'mean_dsc': mean_dice,
        'model_state': model.state_dict(),
        'history_loss_trn': history_loss_trn,
        'history_loss_vld': history_loss_vld,
        'history_DSC_trn': history_DSC_trn,
        'history_DSC_vld': history_DSC_vld,
    }, save_path)

    print(f"model is saved in {save_path}")

# -----------------------------------------TRAIN NEW-------------------------------
def train(train_loader,
                optimizer=None,
                model=None,
                val_loader=None,
                scheduler=None,
                model_save_path = None,
                start_epoch = 0,
                device = None,
                max_epochs = 0,
                dice_loss = None,
                bce = None,
                mse = None,
                window_infer = None,
                best_mean_dice = 0.0,
                args = None
              ):
    
    if model is not None:
        print(f"check model parameter: {next(model.parameters()).sum()}")
        para = sum([np.prod(list(p.size())) for p in model.parameters()])
        print(f"model parameters is {para * 4 / 1000 / 1000}M ")
        model.to(device)
        #os.makedirs(args.logdir, exist_ok=True)

    history_loss = {"train": [], "val": []}
    history_accuracy = {"train": [], "val": []}
    for epoch in range(start_epoch, max_epochs + start_epoch):
        sum_batch_loss = {"train": 0, "val": 0}
        sum_batch_dice = {"train": 0, "val": 0}
        #----------------TRAIN EPOCH------------------------
        if model is not None:
            model.train()

        with tqdm(total=len(train_loader)) as tr:

            for idx, batch in enumerate(train_loader):
                tr.set_description('Epoch %i' % epoch)
                if isinstance(batch, dict):
                    batch = {
                        x: batch[x].contiguous().to(device)
                        for x in batch if isinstance(batch[x], torch.Tensor)
                    }
                elif isinstance(batch, list) :
                    batch = [x.to(device) for x in batch if isinstance(x, torch.Tensor)]

                elif isinstance(batch, torch.Tensor):
                    batch = batch.to(device)
                
                else :
                    print("not support data type")
                    exit(0)
                
                if model is not None:
                    for param in model.parameters(): param.grad = None
                
                #----------------------TRAINING STEP---------------------
                image, label = batch["images"], batch["mask"]
                    
                label = label.float()
                
                x_start = label
                x_start = (x_start) * 2 - 1
                x_t, t, noise = model(x=x_start, pred_type="q_sample", args=args)
                pred_xstart = model(x=x_t, step=t, image=image, pred_type="denoise", args=args)
                #-------LOSS TRAIN--------
                loss_dice = dice_loss(pred_xstart, label)
                if bce is not None:
                    loss_bce = bce(pred_xstart, label)
                pred_xstart = torch.sigmoid(pred_xstart)
                if mse is not None:
                    loss_mse = mse(pred_xstart, label)
                if (mse and bce) is not None:
                    loss = loss_dice + loss_bce + loss_mse
                else:
                    loss = loss_dice
                sum_batch_loss["train"] += loss.item()
                
                #-------DICE TRAIN--------
                output = (pred_xstart > 0.5).float().cpu().numpy()
                target = label.cpu().numpy()
                o = output[:, 0]
                t = target[:, 0]
                ms = dice(o, t)
                sum_batch_dice["train"] += ms#.item() 
                #----------------------END TRAINING STEP---------------------
                #AGGIORNO I GRADIENTI
                loss.backward()
                optimizer.step()
                lr = optimizer.state_dict()['param_groups'][0]['lr']
                tr.set_postfix(loss=loss.item(), lr=lr)
                tr.update(1)
            
            for param in model.parameters() : param.grad = None

            #----------------END TRAIN EPOCH------------------------
            #VALIDATION OGNI TOT EPOCHE
        if (epoch+1) % args.val_every == 0 and val_loader is not None :
            val_outputs = []
            model.eval()
            for idx, batch in tqdm(enumerate(val_loader), total=len(val_loader)):
                if isinstance(batch, dict):
                    batch = {
                        x: batch[x].to(device)
                        for x in batch if isinstance(batch[x], torch.Tensor)
                    }
                elif isinstance(batch, list) :
                    batch = [x.to(device) for x in batch if isinstance(x, torch.Tensor)]

                elif isinstance(batch, torch.Tensor):
                    batch = batch.to(device)
                
                else :
                    print("not support data type")
                    exit(0)
                with torch.no_grad():
                    #--------------------------VALIDATION STEP-----------------------
                    image, label = batch["images"], batch["mask"]
                    
                    label = label.float()
                    
                    output = window_infer(image, model, pred_type="ddim_sample", args=args)
                    
                    #-------LOSS VAL--------
                    loss_dice = dice_loss(output, label)
                    if bce is not None:
                        loss_bce = bce(output, label)
                    output = torch.sigmoid(output)
                    if mse is not None:
                        loss_mse = mse(output, label)
                    if (mse and bce) is not None:
                        loss = loss_dice + loss_bce + loss_mse #LOSS VAL
                    else:
                        loss = loss_dice
                    sum_batch_loss["val"] += loss.item()
                    #-------DICE VAL--------
                    output = (output > 0.5).float().cpu().numpy()
                    target = label.cpu().numpy()
                    o = output[:, 0]
                    t = target[:, 0]
                    ms = dice(o, t)

                    sum_batch_dice["val"] += ms#.item()
                    
                    #--------------------------END VALIDATION STEP-----------------------
                    #SAVE IMAGE
                    '''
                    img_to_save= view_sample_prediction(image, output, target)
                    for id, image in enumerate(imgs_to_save): ###
                        imgs_fname = 'output-batch%d-img%d-epoch%d.png' % (idx, id, epoch) ###
                        imgs_fpath = os.path.join(model_save_path, imgs_fname) ###
                        save_image(image, imgs_fpath, nrow=5) ###
                    '''
                
                val_outputs.append(ms)
            val_outputs = torch.tensor(val_outputs)
            sum_ms = 0
            for i, v in enumerate(val_outputs):
                print(f"Dice ms {i}: ", v)
                sum_ms+=v
            print(f"Dice mean ms is {sum_ms/len(val_outputs)}")
            #------------------------------VALIDATION END-----------------------
            #CALCOLIAMO I VALORI DI LOSS E DICE OGNI VAL_EPOCH, tab in meno di questo blocco per calcolarli a ogni epoca
            # Compute epoch loss/accuracy
            epoch_loss = {split: sum_batch_loss[split]/(len(train_loader) if split=="train" else len(val_loader)) for split in ["train", "val"]}
            epoch_accuracy = {split: sum_batch_dice[split]/(len(train_loader) if split=="train" else len(val_loader)) for split in ["train", "val"]}

            for split in ["train", "val"]:
                history_loss[split].append(epoch_loss[split])
                history_accuracy[split].append(epoch_accuracy[split])

            print("-" * 100)
            print("Epoch {}/{}".format(epoch + start_epoch, max_epochs + start_epoch))
            print(f"TRAIN LOSS: {epoch_loss['train']:.4f}")
            print(f"VAL LOSS: {epoch_loss['val']:.4f}")
            print(f"TRAIN DICE: {epoch_accuracy['train']:.4f}")
            print(f"VAL DICE: {epoch_accuracy['val']:.4f}")

            mean_dice = epoch_accuracy['val']

            if mean_dice > best_mean_dice:
                best_mean_dice = mean_dice
                save_new_model_and_delete_last(model, epoch, mean_dice, history_loss["train"], history_loss["val"], history_accuracy["train"], history_accuracy["val"],
                                                os.path.join(model_save_path, 
                                                f"best_model_{mean_dice:.4f}.pt"), 
                                                delete_symbol="best_model")

            save_new_model_and_delete_last(model, epoch, mean_dice, history_loss["train"], history_loss["val"], history_accuracy["train"], history_accuracy["val"],
                                            os.path.join(model_save_path, 
                                            f"final_model_{mean_dice:.4f}.pt"), 
                                            delete_symbol="final_model")
        
        #------------------------------END VALIDATION END-----------------------
        if scheduler is not None:
            scheduler.step()
        if model is not None:
            model.train()
    # Plot loss history
    plt.title("Loss, Epoche: "+ str(len(history_loss['train'])))
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(history_loss['train'], label='train')
    plt.plot(history_loss['val'], label='val')
    plt.legend()
    plt.savefig(os.path.join(model_save_path, 'Loss_numepoche_{}.png'.format(max_epochs)))
    plt.close()

    # Plot DSC history
    plt.title("DSC, Epoche: "+ str(len(history_accuracy['val'])))
    plt.xlabel('Epoch')
    plt.ylabel('DSC')
    plt.plot(history_accuracy['train'], label='train')
    plt.plot(history_accuracy['val'], label='val')
    plt.legend()
    plt.savefig(os.path.join(model_save_path, 'DSC_numepoche_{}.png'.format(max_epochs)))
    plt.close()

#------------------------------------------TEST---------------------------------------------------
def validation_single_gpu(val_loader, model, device, window_infer, save_path, args):
    model.to(device)
    val_outputs = []
    model.eval()
    for idx, batch in tqdm(enumerate(val_loader), total=len(val_loader)):
        start_time = time.time()
        img_name = batch["images_meta_dict"]["filename_or_obj"][0].split("/")[-1].split("_FLAIR.nii.gz")[0]
        original_affine = batch["images_meta_dict"]["original_affine"][0].numpy()
        
        if isinstance(batch, dict):
            batch = {
                x: batch[x].to(device)
                for x in batch if isinstance(batch[x], torch.Tensor)
            }
        elif isinstance(batch, list) :
            batch = [x.to(device) for x in batch if isinstance(x, torch.Tensor)]

        elif isinstance(batch, torch.Tensor):
            batch = batch.to(device)
        
        else :
            print("not support data type")
            exit(0)

        with torch.no_grad():
            #---------------VALIDATION STEP------------
            
            image = batch["images"]
            label = batch["mask"]
          

            pred_name = img_name + "_PRED.nii.gz"
            label_name = img_name + "_GT.nii.gz"

            label = label.float()

            #----INFERENZA----
            print("Inference on case {}".format(img_name))
            output = window_infer(image, model, pred_type="ddim_sample", args=args)
            nib.save(nib.Nifti1Image(image.squeeze()[0], original_affine), os.path.join(save_path, img_name+".nii.gz"))
            
            output = torch.sigmoid(output)
            output = (output > 0.5).float().cpu().numpy()

            target = label.cpu().numpy()
            output_sq = output.squeeze()
            val_label = target.squeeze()

            o = output[:, 0]
            t = target[:, 0]
            
            #-----METRICS-----
            ms_dice = dice(o, t)
            ms_hd = hausdorff_distance_95(o, t)
            ms_recall = recall(o, t)
            ms_ppv = precision(o,t)
            ms_fpr = false_positive_rate(o, t)
            ms_ltpr = ltpr(o, t)
            ms_lfpr = lfpr(o, t)
            ms_avd = compute_avd(t, o)
            ms_assd = avg_surface_distance_symmetric(o, t)


            #img_to_save= view_sample_prediction(image, output, target)
            print(f"DICE: {ms_dice}")
            print(f"TPR (Recall,sensitivity, TPR): {ms_recall}")
            print(f"PPV (precision): {ms_ppv}")
            print(f"FPR (1-specificity) is {ms_fpr}")
            print(f"HD: {ms_hd}")
            print(f"LTPR: {ms_ltpr}")
            print(f"LFPR: {ms_lfpr}")
            print(f"AVD: {ms_avd}")
            print(f"ASSD: {ms_assd}")

            val_out=[ms_dice, ms_recall, ms_ppv, ms_fpr, ms_hd, ms_ltpr, ms_lfpr, ms_avd, ms_assd]
            #---------------END VALIDATION STEP------------
            nib.save(nib.Nifti1Image(output_sq.astype(np.uint8), original_affine), os.path.join(save_path, pred_name))
            nib.save(nib.Nifti1Image(val_label.astype(np.uint8), original_affine), os.path.join(save_path, label_name))
            '''
            for id, image in enumerate(img_to_save): ###
                imgs_fname = img_name + '-batch%d-img%d.png' % (idx, id) ###
                imgs_fpath = os.path.join(save_path, imgs_fname) ###
                save_image(image, imgs_fpath, nrow=5) ###
            '''
            end_time = time.time()
        print(f"Time for 1 batch: {end_time-start_time} seconds")
        val_outputs.append(val_out)
    val_outputs = torch.tensor(val_outputs)

    num_val = len(val_outputs[0])
    length = [0.0 for i in range(num_val)]
    v_sum = [0.0 for i in range(num_val)]

    for v in val_outputs:
        for i in range(num_val):
            if not torch.isnan(v[i]):
                v_sum[i] += v[i]
                length[i] += 1

    for i in range(num_val):
        if length[i] == 0:
            v_sum[i] = 0
        else :
            v_sum[i] = v_sum[i] / length[i]
    return v_sum, val_outputs 

def view_sample_prediction(input, np_pred, target):
        imgs_to_save = []
        input = input.squeeze(0).cuda()
        np_pred= np.squeeze(np_pred,0)
        target= np.squeeze(target,0)
        np_pred = torch.from_numpy(np_pred).cuda()
        target = torch.from_numpy(target).cuda()

        for t in range(target.shape[1]-1): #target.shape[1] = 140
            #if t == 86:
                #print("t", t)
                img_to_save = []

                notditectedlesion=np.zeros(np_pred.shape)
                wronglesiondetected=np.zeros(np_pred.shape)

                notditectedlesion[:,t] = torch.logical_and(target[:,t].cpu()==1, np_pred[:,t].cpu()==0)
                wronglesiondetected[:,t] = torch.logical_and(target[:,t].cpu()==0, np_pred[:,t].cpu()==1)

                np_pred = torch.from_numpy(ndi.binary_fill_holes(np_pred.cpu()).astype(int)).cuda()
                false_negative_mask = torch.from_numpy(ndi.binary_fill_holes(notditectedlesion).astype(int))
                false_positive_mask = torch.from_numpy(ndi.binary_fill_holes(wronglesiondetected).astype(int))
                np_pred = np_pred.float()
                false_negative_mask = false_negative_mask.float()
                false_positive_mask = false_positive_mask.float()
                
                input_save = input[2,:,:,t].unsqueeze(0)
                img_to_save.append(torch.cat((input_save, input_save, input_save), 0).cuda()) #torch.Size([3, 160, 150]) ->OK!
                img_to_save.append(torch.cat((target[:,:,:,t], target[:,:,:,t], target[:,:,:,t]), 0).cuda()) #torch.Size([3, 160, 150])
                img_to_save.append(torch.cat((np_pred[:,:,:,t], np_pred[:,:,:,t], np_pred[:,:,:,t]), 0).cuda())#torch.Size([3, 160, 150])
                img_to_save.append(torch.cat((false_negative_mask[:,:,:,t], false_negative_mask[:,:,:,t], false_negative_mask[:,:,:,t]), 0).cuda()) #torch.Size([3, 160, 150])
                img_to_save.append(torch.cat((false_positive_mask[:,:,:,t], false_positive_mask[:,:,:,t], false_positive_mask[:,:,:,t]), 0).cuda()) #torch.Size([3, 160, 150])

                imgs_to_save.append(img_to_save)
        return imgs_to_save

            