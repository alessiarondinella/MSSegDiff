from unet.basic_unet_denose import BasicUNetDe
from unet.basic_unet import BasicUNetEncoder
from unet.SegResNet import SegResNet
from unet.SegResNet_denose import SegResNetDe
from unet.SwinUNETR import SwinUNETR
from unet.SwinUNETR_denose import SwinUNETRDe

from guided_diffusion.gaussian_diffusion import get_named_beta_schedule, ModelMeanType, ModelVarType,LossType
from guided_diffusion.respace import SpacedDiffusion, space_timesteps
from guided_diffusion.resample import UniformSampler
import torch
import torch.nn as nn 
from guided_diffusion import unet
import matplotlib.pyplot as plt
import os
from torchvision.utils import save_image
from guided_diffusion.utils import staple
import SimpleITK as sitk
import numpy as np

class MSSegDiff(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.number_targets = args.number_targets

        if(args.model == 'MSSegDiff'):
            self.embed_model = BasicUNetEncoder(3, args.number_modality, args.number_targets, args.feature)
            self.model = BasicUNetDe(3, args.number_modality + args.number_targets, args.number_targets, args.feature, 
                                    act = ("LeakyReLU", {"negative_slope": 0.1, "inplace": False}))
        
        elif(args.model == 'MSSegDiff+EncoderSegResNet'):
            self.embed_model = SegResNet(spatial_dims = 3, in_channels= args.number_modality, out_channels=args.number_targets, blocks_down=(1, 2, 2, 4), init_filters=64, num_groups=64, norm = ("GROUP", {"num_groups": 64}))
            self.model = SegResNetDe(spatial_dims = 3, in_channels=args.number_modality+args.number_targets, out_channels=args.number_targets, blocks_down=(1, 2, 2, 4), init_filters=64, num_groups=64, norm = ("GROUP", {"num_groups": 64}))

        elif(args.model == 'MSSegDiff+SegResNet'):
            self.embed_model = SegResNet(spatial_dims = 3, in_channels= args.number_modality, out_channels=args.number_targets, blocks_down=(1, 2, 2, 4), init_filters=64, num_groups=64, norm = ("GROUP", {"num_groups": 64}))
            self.model = SegResNetDe(spatial_dims = 3, in_channels=args.number_modality+args.number_targets, out_channels=args.number_targets, blocks_down=(1, 2, 2, 4), init_filters=64, num_groups=64, norm = ("GROUP", {"num_groups": 64}))

        elif(args.model == 'MSSegDiff+SwinUNETR'):
            self.embed_model = SwinUNETR(img_size=(96,96,96), in_channels=args.number_modality, out_channels=args.number_targets, feature_size = 48, use_checkpoint=True) #12
            self.model = SwinUNETRDe(img_size=(96,96,96), in_channels=args.number_modality+args.number_targets, out_channels=args.number_targets,feature_size = 48, use_checkpoint=True)#12
        
        elif(args.model == 'MSSegDiff+MultiEncoder'):
            self.embed_model = BasicUNetEncoder(3, 1, args.number_targets, args.feature)#[64, 64, 128, 256, 512, 64])#, dropout=0.5)
            self.model = BasicUNetDe(3, args.number_modality + args.number_targets, args.number_targets, args.feature, 
                                    act = ("LeakyReLU", {"negative_slope": 0.1, "inplace": False}))

        betas = get_named_beta_schedule("linear", 1000)
        self.diffusion = SpacedDiffusion(use_timesteps=space_timesteps(1000, [1000]),
                                            betas=betas,
                                            model_mean_type=ModelMeanType.START_X,
                                            model_var_type=ModelVarType.FIXED_LARGE,
                                            loss_type=LossType.MSE,
                                            )

        self.sample_diffusion = SpacedDiffusion(use_timesteps=space_timesteps(1000, [50]),
                                            betas=betas,
                                            model_mean_type=ModelMeanType.START_X,
                                            model_var_type=ModelVarType.FIXED_LARGE,
                                            loss_type=LossType.MSE,
                                            )
        self.sampler = UniformSampler(1000)


    def forward(self, image=None, x=None, pred_type=None, step=None, args=None):
        #ADD NOISE
        if pred_type == "q_sample":
            noise = torch.randn_like(x).to(x.device)
            t, weight = self.sampler.sample(x.shape[0], x.device)
            return self.diffusion.q_sample(x, t, noise=noise), t, noise
        #DENOISE
        elif pred_type == "denoise":
            if(args.model == "MSSegDiff+MultiEncoder"):
                embedding1 = self.embed_model(image[:,0].unsqueeze(1)) # torch.Size([2, 1, 96, 96, 96])
                embedding2 = self.embed_model(image[:,1].unsqueeze(1)) # torch.Size([2, 1, 96, 96, 96])
                embedding3 = self.embed_model(image[:,2].unsqueeze(1)) # torch.Size([2, 1, 96, 96, 96])
               
                embeddings = [embedding1[x]+embedding2[x]+embedding3[x] for x in range(len(embedding1))]
            else:
                embeddings = self.embed_model(image)         
            
            return self.model(x, t=step, image=image, embeddings=embeddings)

        elif pred_type == "ddim_sample":
            if(args.model == "MSSegDiff+MultiEncoder"):
                embedding1 = self.embed_model(image[:,0].unsqueeze(1)) # torch.Size([2, 1, 96, 96, 96])
                embedding2 = self.embed_model(image[:,1].unsqueeze(1)) # torch.Size([2, 1, 96, 96, 96])
                embedding3 = self.embed_model(image[:,2].unsqueeze(1)) # torch.Size([2, 1, 96, 96, 96])
               
                embeddings = [embedding1[x]+embedding2[x]+embedding3[x] for x in range(len(embedding1))]
            else:
                embeddings = self.embed_model(image)  
                       
            sample_out = self.sample_diffusion.ddim_sample_loop(self.model, (1, self.number_targets, 96, 96, 96), model_kwargs={"image": image, "embeddings": embeddings})
            sample_out = sample_out["pred_xstart"]
            return sample_out
        
#---------------------------------------------TEST----------------------------------------------        
def compute_uncer(pred_out):
    pred_out = torch.sigmoid(pred_out)
    pred_out[pred_out < 0.001] = 0.001
    uncer_out = - pred_out * torch.log(pred_out)
    return uncer_out
    

class MSSegDiff_test(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.s = args.s
        self.configuration_test = args.configuration_test        
        self.number_targets = args.number_targets

        if(args.model == 'MSSegDiff'):
            self.embed_model = BasicUNetEncoder(3, args.number_modality, args.number_targets, args.feature)
            self.model = BasicUNetDe(3, args.number_modality + args.number_targets, args.number_targets, args.feature, 
                                    act = ("LeakyReLU", {"negative_slope": 0.1, "inplace": False}))
        
        elif(args.model == 'MSSegDiff+EncoderSegResNet'):
            self.embed_model = SegResNet(spatial_dims = 3, in_channels= args.number_modality, out_channels=args.number_targets, blocks_down=(1, 2, 2, 4), init_filters=64, num_groups=64, norm = ("GROUP", {"num_groups": 64}))
            self.model = SegResNetDe(spatial_dims = 3, in_channels=args.number_modality+args.number_targets, out_channels=args.number_targets, blocks_down=(1, 2, 2, 4), init_filters=64, num_groups=64, norm = ("GROUP", {"num_groups": 64}))

        elif(args.model == 'MSSegDiff+SegResNet'):
            self.embed_model = SegResNet(spatial_dims = 3, in_channels= args.number_modality, out_channels=args.number_targets, blocks_down=(1, 2, 2, 4), init_filters=64, num_groups=64, norm = ("GROUP", {"num_groups": 64}))
            self.model = SegResNetDe(spatial_dims = 3, in_channels=args.number_modality+args.number_targets, out_channels=args.number_targets, blocks_down=(1, 2, 2, 4), init_filters=64, num_groups=64, norm = ("GROUP", {"num_groups": 64}))

        elif(args.model == 'MSSegDiff+SwinUNETR'):
            self.embed_model = SwinUNETR(img_size=(96,96,96), in_channels=args.number_modality, out_channels=args.number_targets, feature_size = 48, use_checkpoint=True) #12
            self.model = SwinUNETRDe(img_size=(96,96,96), in_channels=args.number_modality+args.number_targets, out_channels=args.number_targets,feature_size = 48, use_checkpoint=True)#12
        
        elif(args.model == 'MSSegDiff+MultiEncoder'):
            self.embed_model = BasicUNetEncoder(3, 1, args.number_targets, args.feature)#[64, 64, 128, 256, 512, 64])#, dropout=0.5)
            self.model = BasicUNetDe(3, args.number_modality + args.number_targets, args.number_targets, args.feature, 
                                    act = ("LeakyReLU", {"negative_slope": 0.1, "inplace": False}))
        
        betas = get_named_beta_schedule("linear", 1000)
        self.diffusion = SpacedDiffusion(use_timesteps=space_timesteps(1000, [1000]),
                                            betas=betas,
                                            model_mean_type=ModelMeanType.START_X,
                                            model_var_type=ModelVarType.FIXED_LARGE,
                                            loss_type=LossType.MSE,
                                            )

        self.sample_diffusion = SpacedDiffusion(use_timesteps=space_timesteps(1000, [10]), #T=10
                                            betas=betas,
                                            model_mean_type=ModelMeanType.START_X,
                                            model_var_type=ModelVarType.FIXED_LARGE,
                                            loss_type=LossType.MSE,
                                            )
        self.sampler = UniformSampler(1000)

    def forward(self, image=None, x=None, pred_type=None, step=None, embedding=None, args=None):
        if pred_type == "q_sample":
            noise = torch.randn_like(x).to(x.device)
            t, weight = self.sampler.sample(x.shape[0], x.device)
            return self.diffusion.q_sample(x, t, noise=noise), t, noise

        elif pred_type == "denoise":
            embeddings = self.embed_model(image)
            
            return self.model(x, t=step, image=image, embedding=embeddings)
            #return self.model(x, timesteps=step, y=image) #UNETMODEL

        elif pred_type == "ddim_sample":
            embeddings = self.embed_model(image)
            
            if self.configuration_test == '_uncertainty':
                #SUF configuration from https://arxiv.org/pdf/2303.10326

                uncer_step = self.s
                
                #print("uncer_step: ", uncer_step)
                sample_outputs = []
                for i in range(uncer_step):
                    sample_outputs.append(self.sample_diffusion.ddim_sample_loop(self.model, (1, self.number_targets, 96, 96, 96), model_kwargs={"image": image, "embeddings": embeddings}))

                sample_return = torch.zeros((1, self.number_targets, 96, 96, 96))
                #print(len(sample_outputs[0]["all_model_outputs"]))
                for index in range(10): #index è il prediction step corrente
    
                    uncer_out = 0
                    for i in range(uncer_step):
                        uncer_out += sample_outputs[i]["all_model_outputs"][index] #uncer_out = fusion weight
                    uncer_out = uncer_out / uncer_step
                    uncer = compute_uncer(uncer_out).cpu()

                    w = torch.exp(torch.sigmoid(torch.tensor((index + 1) / 10)) * (1 - uncer))
                
                    for i in range(uncer_step):
                        sample_return += w * sample_outputs[i]["all_samples"][index].cpu()
                        
            elif self.configuration_test == '_mean':                   
                #RESTITUISCO TUTTI I SAMPLE DEL PREDICTION STEP (sample_out è un dict)
                testdir = "/storage/data_4T/alessiarondinella_data/lesion_volume/logs_PROVE/"
                sample_out = self.sample_diffusion.ddim_sample_loop(self.model, (1, self.number_targets, 96, 96, 96), model_kwargs={"image": image, "embeddings": embeddings})
                #CREO TENSORI VUOTI CHE CONTERRANNO IL RISULTATO
                sample_return = torch.zeros((1, self.number_targets, 96, 96, 96))
                sample_return_all_model_output = torch.zeros((1, self.number_targets, 96, 96, 96))

                for index in range(10): #index è il prediction step corrente
                    sample_return += sample_out["all_samples"][index].cpu() #torch.Size([1, 1, 96, 96, 96])
                
                #ALL SAMPLES
                lst = torch.stack(sample_out["all_samples"]) #lst size = #torch.Size([10, 1, 1, 96, 96, 96])
                mean = torch.mean(lst, 0) #mean size = #torch.Size([1, 1, 96, 96, 96])
                var = torch.var(lst, 0)
                std=torch.std(lst, 0)
                
                #ALL MODEL OUTPUT
                lst_all_model_output = torch.stack(sample_out["all_model_outputs"])
                mean_all_model_output = torch.mean(lst_all_model_output, 0) #mean size = #torch.Size([1, 1, 96, 96, 96])
                var_all_model_output = torch.var(lst_all_model_output, 0)
                std_all_model_output=torch.std(lst_all_model_output, 0)

                sample_return = mean # ritorno la media di 10 campioni all_sample

            elif self.configuration_test == '_mean-var':
                sample_out = self.sample_diffusion.ddim_sample_loop(self.model, (1, self.number_targets, 96, 96, 96), model_kwargs={"image": image, "embeddings": embeddings})
                sample_return = torch.zeros((1, self.number_targets, 96, 96, 96))  

                for index in range(10): #index è il prediction step corrente
                    sample_return += sample_out["all_samples"][index].cpu() #torch.Size([1, 1, 96, 96, 96])

                #ALL SAMPLES
                lst = torch.stack(sample_out["all_samples"]) #lst size = #torch.Size([10, 1, 1, 96, 96, 96])
                mean = torch.mean(lst, 0) #mean size = #torch.Size([1, 1, 96, 96, 96])
                var = torch.var(lst, 0)
                std=torch.std(lst, 0)

                sample_return = mean-var

            elif self.configuration_test == '_staple':
                sample_out = self.sample_diffusion.ddim_sample_loop(self.model, (1, self.number_targets, 96, 96, 96), model_kwargs={"image": image, "embeddings": embeddings})
                #print(torch.stack(sample_out["all_samples"],dim=0).size()) #torch.Size([10, 1, 1, 96, 96, 96])
                enslist = []
                for index in range(10):
                    #enslist.append(sample_out["all_samples"][index].cpu().squeeze())
                    enslist.append(sitk.GetImageFromArray(sample_out["all_samples"][index].cpu().numpy().squeeze().astype(np.int16)))
                
                staple_seg = sitk.STAPLE(enslist, 1.0)
                sample_return = sitk.GetArrayFromImage(staple_seg)
                sample_return = torch.tensor(sample_return).unsqueeze(0)
                sample_return = sample_return.unsqueeze(0)
                #print(sample_return.size())
            
            return sample_return
