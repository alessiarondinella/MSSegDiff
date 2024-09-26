
import os 
import glob 
import torch 

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
