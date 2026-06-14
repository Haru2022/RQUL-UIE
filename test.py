import numpy as np
import os
from PIL import Image
from diffusers import UNet2DConditionModel
from diffusers import UNet2DConditionModel,DDIMScheduler,DDPMScheduler
from WaterQuantNoisePipeline_v1 import WaterQuantNoisePipeline 
import torch
from fft_refiner_0408 import FFT_Refiner
from collections import OrderedDict
from transformers import CLIPTextModel, CLIPTokenizer





unet = UNet2DConditionModel.from_pretrained("HaruCloud9/RQUL-UIE",
                                            subfolder="unet",
                                            torch_dtype=torch.float32,
                                            )

#unet.enable_xformers_memory_efficient_attention()
scheduler = DDPMScheduler.from_pretrained(
            "sd2-community/stable-diffusion-2-1", 
            subfolder="scheduler", 
            timestep_spacing="trailing", # set scheduler timestep spacing to trailing for later inference.
        )

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

fft_refiner_path = 'final_model.pth'
fft_refiner_state_dict = torch.load(fft_refiner_path, map_location='cpu')
new_state_dict = OrderedDict()
for k, v in fft_refiner_state_dict.items():
    name = k[7:] if k.startswith('module.') else k
    new_state_dict[name] = v
fft_refiner = FFT_Refiner()
fft_refiner.load_state_dict(new_state_dict)
fft_refiner.to(device, dtype=torch.float32)


pipeline = WaterQuantNoisePipeline.from_pretrained("sd2-community/stable-diffusion-2-1",
                                                    unet = unet,
                                                    scheduler=scheduler,
                                                    torch_dtype=torch.float32,
                                                    ).to(device)
datasets = ['EUVP','LSUI','haru_1473']

datasets = datasets[::-1]

for dataset in datasets:
    test_root = '/media/HDD0/haru/datasets/water-mamba/{}/raw'.format(dataset)
    save_root = '/media/HDD2/haru/Datasets/WaterDecouple/benchmark_result/{}/RQUL-UIE'.format(dataset) 


    if not os.path.exists(save_root):
        os.makedirs(save_root, exist_ok=True)

    img_names = os.listdir(test_root)
    img_names = [name for name in img_names if name.endswith('.jpg') or name.endswith('.png')]
    img_names = sorted(img_names)
    img_names = img_names[::-1]

    img_size = 512
    save_size = 256

    def resize_image(image:Image.Image, size):
        return image.resize((size, size), Image.LANCZOS)

    with torch.no_grad():
        total_inference_time = 0.0 
        measured_batches = 0      
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        for img_name in img_names:
            #if not img_name.endswith('.png') or not img_name.endswith('.jpg'):
            #    continue
            image_path = os.path.join(test_root, img_name)
            raw_img = Image.open(image_path).convert("RGB")
            img = resize_image(raw_img, img_size)
            starter.record()
            out = pipeline(
                input_image=img,
                height=img_size,
                width=img_size,
                fft_refiner=fft_refiner,
                max_noise_level=9,
                fixed_single_level=9,
            )
            ender.record()
            torch.cuda.synchronize()
            inference_time = starter.elapsed_time(ender)
            total_inference_time += inference_time
            measured_batches += 1
            res_pil = out.refine_res_pil
            #res_pil = out.coarse_res_pil
            res_pil = res_pil.resize((save_size, save_size), Image.LANCZOS)
            #res_pil = res_pil.resize(raw_img.size, Image.LANCZOS)
            save_name = os.path.splitext(img_name)[0] + '.png'
            res_pil.save(os.path.join(save_root, save_name))
            print(f"Processed {img_name}, inference time: {inference_time:.2f} ms")

        if measured_batches > 0:
            avg_inference_time = total_inference_time / measured_batches
            print(f"Average inference time for {dataset}: {avg_inference_time:.2f} ms")