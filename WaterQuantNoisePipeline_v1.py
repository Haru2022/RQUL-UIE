from typing import Any, Dict, Union

import torch.nn as nn
import torch
#from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm.auto import tqdm
from PIL import Image
from diffusers import (
    DiffusionPipeline,
    DDIMScheduler,
    AutoencoderKL,
    UNet2DConditionModel,
)
#from models.unet2dconditionwater import UNet2DConditionModel
#from ..models.unet_2d_condition import UNet2DConditionModel
from diffusers.utils import BaseOutput
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection, CLIPTokenizer,CLIPTextModel
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
from fft_refiner_0408 import FFT_Refiner
from PIL import Image
#from util.physical_encoder import physical_encoder
import warnings
#from util.WaterPhysicalLosses_v3 import disp_to_depth, depth_to_disp

import cv2

def profile_module(module, inputs):
    """统计 FLOPs 和显存"""
    device = "cuda"
    module = module.to(device).eval()
    inputs = tuple(i.to(device) for i in inputs)
    from fvcore.nn import FlopCountAnalysis

    # 参数量
    params = sum(p.numel() for p in module.parameters())

    # FLOPs
    try:
        analysis = FlopCountAnalysis(module, inputs)
        analysis = analysis.unsupported_ops_warnings(False)
        flops = analysis.total()
    except Exception as e:
        print(f"FLOPs 统计失败: {e}")
        flops = None

    # 显存
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        _ = module(*inputs)
    mem = torch.cuda.max_memory_allocated()

    return params, flops, mem

class WaterQuantNoisePipelineOutput(BaseOutput):
    air_np_coarses: Union[list[np.ndarray], np.ndarray] = None
    air_pil_coarses: Union[list[Image.Image], Image.Image] = None
    air_np_refines: Union[list[np.ndarray], np.ndarray] = None
    air_pil_refines: Union[list[Image.Image], Image.Image] = None
    #res_tensor: torch.Tensor

class WaterQuantNoisePipeline(DiffusionPipeline):
    
    latent_scale_factor = 0.18215
    
    def __init__(self,
                 unet:UNet2DConditionModel,
                 vae:AutoencoderKL,
                 scheduler:DDIMScheduler,
                 tokenizer:CLIPTokenizer,
                 text_encoder:CLIPTextModel,
                 ):
        super().__init__()
            
        self.register_modules(
            unet=unet,
            vae=vae,
            scheduler=scheduler,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
        )
        
        self.img_embed = None 
        self.task_embed = None
        self.pre_latents = None
        self.pre_latents_ratio = 0.1
        self.fft_refiner = None
        
    # Apply VAE Encoder to image   
    def encode_RGB(self, image):
        h = self.vae.encoder(image)
        moments = self.vae.quant_conv(h)
        latent, _ = torch.chunk(moments, 2, dim=1)
        latent = latent * self.latent_scale_factor
        return latent

    # Apply VAE Decoder to latent
    def decode_RGB(self, latent):
        latent = latent / self.latent_scale_factor
        z = self.vae.post_quant_conv(latent)
        image = self.vae.decoder(z)
        return image
    
        
    def task_prompt_encode(self):
        """
        Encode text embedding for empty prompt
        """

        prompt = ""
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            #max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(self.text_encoder.device)
        if isinstance(self.unet, UNet2DConditionModel):
            self.task_embed = self.text_encoder(text_input_ids,return_dict=False)[1].to(self.dtype)
        else:
            raise NotImplementedError
            self.task_embed = self.text_encoder(text_input_ids)[1].to(self.dtype)
            
    def input_preprocess(self, input_image:Union[Image.Image, np.ndarray], height=None, width=None):
        assert isinstance(input_image, (Image.Image, np.ndarray)), "input_image should be PIL Image or np.ndarray"
        if isinstance(input_image, np.ndarray): # only for sequential inference, no resize operation
            out = torch.from_numpy(input_image.copy()).to(self.dtype).to(self.device)
            if out.dim() == 2:
                out = out.unsqueeze(0).repeat(3,1,1)
            else:
                out = out.permute((2,0,1)).contiguous()
            out = out*2.0 - 1.0 # [0, 1] -> [-1, 1]
        elif isinstance(input_image, Image.Image):
            input_width, input_height = input_image.size
            if height != input_height or width != input_width:
                print("resize")
                input_image = input_image.resize((width, height))
                
            if input_image.mode == "RGB":
                image = np.array(input_image)
                out = np.transpose(image, (2, 0, 1))
                out = out / 255.0 * 2.0 - 1.0 # [0, 255] -> [-1, 1]
            else:
                image = np.array(input_image)
                image = (image[:, :, np.newaxis]/65535.0) *2.0 -1.0 # [0, 65535] -> [-1, 1]
                out = image.repeat(3,axis=2).transpose(2, 0, 1)
            out = torch.from_numpy(out).to(self.dtype).to(self.device)
        out = out.unsqueeze(0)
        return out
        
    def output_postprocess(self, output:torch.Tensor):
        output = output.clamp(-1.0, 1.0)
        output = (output+1.0) / 2.0
        output_np = output.cpu().numpy().transpose(1, 2, 0)
        if output_np.shape[2] == 1:
            #print(output_np.shape)
            #output_np = np.tile(output_np, (1, 1, 3))
            output_pil = Image.fromarray((output_np * 65535).astype(np.uint16),mode='I;16')
        else:
            output_pil = Image.fromarray((output_np * 255).astype(np.uint8))
        return output_np, output_pil
    
    
    def single_unet_proc(self, rgb_latent, timesteps, batch_task_embed, rgb):
        latent = torch.zeros_like(rgb_latent).to(self.dtype)
        iterable = enumerate(timesteps)
        for i, t in iterable:
            
            #unet_input = rgb_latent if i == 0 else latent
            unet_input = torch.cat([rgb_latent, latent], dim=1) 
            # predict the noise residual
            noise_pred = self.unet(
                unet_input, t, encoder_hidden_states=batch_task_embed).sample  # [B, 4, h, w]

            # compute the previous noisy sample x_t -> x_t-1
            scheduler_step = self.scheduler.step(
                noise_pred, t, latent
            )
            
            
            latent = scheduler_step.prev_sample

        torch.cuda.empty_cache()
        res_coarse = self.decode_RGB(latent)

        coarse_np, coarse_pil = self.output_postprocess(res_coarse.squeeze(0))
        fft_refiner_input = torch.cat([rgb, res_coarse], dim=1)
        res_refine = self.fft_refiner(fft_refiner_input)
        refine_np, refine_pil = self.output_postprocess(res_refine.squeeze(0))
 
        return coarse_np, coarse_pil, refine_np, refine_pil, latent
                

    @torch.no_grad()
    def __call__(self,
                 input_image: Union[Image.Image, np.ndarray],
                 denoising_steps: int = 1,
                 height:int = None,
                 width:int = None,
                 max_noise_level:int = 9,
                 fixed_single_level: int = None,
                 fft_refiner: FFT_Refiner = None
                 ) -> WaterQuantNoisePipelineOutput:
        
        # inherit from thea Diffusion Pipeline
        device = self.device
        self.img_embed = None
        if self.fft_refiner is None:
            self.fft_refiner = fft_refiner.to(self.device).to(dtype=torch.float32)
        rgb = self.input_preprocess(input_image, height, width)
        
        # param, flop, mem = profile_module(self.vae.encoder, (rgb,))
        # print(f"{'VAE Encoder':20s} | {param/1e6:10.2f} | {None if flop is None else flop/1e9:10.2f} | {mem/1024**2:10.2f}")
        
        rgb_latent = self.encode_RGB(rgb)
        

        self.scheduler.set_timesteps(denoising_steps, device=device) # here the numbers of the steps is only 10.
        timesteps = self.scheduler.timesteps  # [T]

        self.task_prompt_encode()
        batch_task_embed = self.task_embed.repeat(
            (1, 1, 1))
        
        air_np_coarses, air_pil_coarses, air_np_refines, air_pil_refines = [], [], [], []
        cur_latent = rgb_latent
        for cur_level in range(1, max_noise_level+1):
            local_timesteps = torch.ones((1,), device=device, dtype=torch.long)*int(cur_level/max_noise_level*timesteps[0])
            air_np_coarse, air_pil_coarse, air_np_refine, air_pil_refine,pre_latent = self.single_unet_proc(cur_latent, 
                                                    local_timesteps, 
                                                    batch_task_embed, 
                                                    rgb,
                                                    )
            cur_latent = pre_latent
            if fixed_single_level is None:
                air_np_coarses.append(air_np_coarse)
                air_pil_coarses.append(air_pil_coarse)
                air_np_refines.append(air_np_refine)
                air_pil_refines.append(air_pil_refine)
            elif fixed_single_level == cur_level:
                air_np_coarses=air_np_coarse
                air_pil_coarses=air_pil_coarse
                air_np_refines=air_np_refine
                air_pil_refines=air_pil_refine


        return WaterQuantNoisePipelineOutput(coarse_res_np=air_np_coarses, coarse_res_pil=air_pil_coarses,
                                        refine_res_np=air_np_refines, refine_res_pil=air_pil_refines)
        
            
            
        
        
        
        
        
        
        
        
        
    