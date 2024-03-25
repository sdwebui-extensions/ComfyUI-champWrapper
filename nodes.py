import argparse
import logging
import os
import os.path as osp
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.utils.checkpoint
from torchvision import transforms
from diffusers import AutoencoderKL, DDIMScheduler
from contextlib import contextmanager, nullcontext
from omegaconf import OmegaConf
from PIL import Image
from transformers import CLIPVisionModelWithProjection

from .models.unet_2d_condition import UNet2DConditionModel
from .models.unet_3d import UNet3DConditionModel
from .models.mutual_self_attention import ReferenceAttentionControl
from .models.guidance_encoder import GuidanceEncoder
from .models.champ_model import ChampModel
import torch.nn.functional as F
from .pipelines.pipeline_aggregation import MultiGuidance2LongVideoPipeline

from .utils.video_utils import resize_tensor_frames, save_videos_grid, pil_list_to_tensor
import comfy.model_management as mm
import folder_paths

def convert_dtype(dtype_str):
    if dtype_str == 'fp32':
        return torch.float32
    elif dtype_str == 'fp16':
        return torch.float16
    elif dtype_str == 'bf16':
        return torch.bfloat16
    else:
        raise NotImplementedError
def setup_savedir(cfg):
    time_str = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    if cfg.exp_name is None:
        savedir = f"results/exp-{time_str}"
    else:
        savedir = f"results/{cfg.exp_name}-{time_str}"
    
    os.makedirs(savedir, exist_ok=True)
    
    return savedir

def setup_guidance_encoder(cfg):
    guidance_encoder_group = dict()
    
    if cfg.weight_dtype == "fp16":
        weight_dtype = torch.float16
    else:
        weight_dtype = torch.float32
    
    for guidance_type in cfg.guidance_types:
        guidance_encoder_group[guidance_type] = GuidanceEncoder(
            guidance_embedding_channels=cfg.guidance_encoder_kwargs.guidance_embedding_channels,
            guidance_input_channels=cfg.guidance_encoder_kwargs.guidance_input_channels,
            block_out_channels=cfg.guidance_encoder_kwargs.block_out_channels,
        ).to(device="cuda", dtype=weight_dtype)
    
    return guidance_encoder_group

def process_semantic_map(semantic_map_path: Path):
    image_name = semantic_map_path.name
    mask_path = semantic_map_path.parent.parent / "mask" / image_name
    semantic_array = np.array(Image.open(semantic_map_path))
    mask_array = np.array(Image.open(mask_path).convert("RGB"))
    semantic_pil = Image.fromarray(np.where(mask_array > 0, semantic_array, 0))
    
    return semantic_pil

script_directory = os.path.dirname(os.path.abspath(__file__))

def combine_guidance_data(cfg, max_files_per_type=None):
    guidance_types = cfg.guidance_types
    guidance_data_folder = os.path.join(script_directory, "example_data", "motions", "motion-01")
    guidance_pil_group = dict()
    for guidance_type in guidance_types:
        guidance_pil_group[guidance_type] = []
        for guidance_image_path in sorted(Path(osp.join(guidance_data_folder, guidance_type)).iterdir()):
            # Add black background to semantic map
            if guidance_type == "semantic_map":
                guidance_pil_group[guidance_type] += [process_semantic_map(guidance_image_path)]
            else:
                guidance_pil_group[guidance_type] += [Image.open(guidance_image_path).convert("RGB")]
            
            # Limit the number of files processed for each guidance type
            if max_files_per_type is not None and len(guidance_pil_group[guidance_type]) >= max_files_per_type:
                break
    
    # get video length from the first guidance sequence
    first_guidance_length = len(list(guidance_pil_group.values())[0])
    # ensure all guidance sequences are of equal length
    assert all(len(sublist) == first_guidance_length for sublist in list(guidance_pil_group.values()))
    
    return guidance_pil_group, first_guidance_length
    


class champ_model_loader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model": ("MODEL",),
            "vae": ("VAE",),
            "ckpt_name": (folder_paths.get_filename_list("checkpoints"), ),
            "diffusion_dtype": (
                    [
                        'fp32',
                        'fp16',
                        'bf16',
                        'auto',
                    ], {
                        "default": 'auto'
                    }),
            "vae_dtype": (
                    [
                        'fp32',
                        'fp16',
                        'bf16',
                        'auto'
                    ], {
                        "default": 'auto'
                    }),
            "fp8_unet": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("CHAMPMODEL", "CHAMPVAE", "CHAMPENCODER")
    RETURN_NAMES = ("champ_model", "champ_vae", "champ_encoder",)
    FUNCTION = "loadmodel"
    CATEGORY = "champWrapper"

    def loadmodel(self, model, vae, diffusion_dtype, vae_dtype, ckpt_name, fp8_unet=False):
        mm.soft_empty_cache()
        device = mm.get_torch_device()
        config_path = os.path.join(script_directory, "configs/inference.yaml")
        cfg = OmegaConf.load(config_path)

        custom_config = {
            'diffusion_dtype': diffusion_dtype,
            'vae_dtype': vae_dtype,
            'ckpt_name': ckpt_name,
            'fp8_unet': fp8_unet,
            'model': model,
            'vae': vae
        }
        if not hasattr(self, 'model') or self.model == None or custom_config != self.current_config:
            self.current_config = custom_config
            # setup pretrained models
            original_config = OmegaConf.load(os.path.join(script_directory, f"configs/v1-inference.yaml"))
            ad_unet_config = OmegaConf.load(os.path.join(script_directory, f"configs/ad_unet_config.yaml"))
            if diffusion_dtype == 'auto':
                try:
                    if mm.should_use_fp16():
                        print("Diffusion using fp16")
                        dtype = torch.float16
                    elif mm.should_use_bf16():
                        print("Diffusion using bf16")
                        dtype = torch.bfloat16
                    else:
                        print("Diffusion using fp32")
                        dtype = torch.float32
                except:
                    raise AttributeError("ComfyUI version too old, can't autodecet properly. Set your dtypes manually.")
            else:
                print(f"Diffusion using {diffusion_dtype}")
                dtype = convert_dtype(diffusion_dtype)

            denoising_unet_path = os.path.join(script_directory,"checkpoints", "denoising_unet.pth")
            reference_unet_path = os.path.join(script_directory,"checkpoints", "reference_unet.pth")
            motion_module_path = os.path.join(script_directory,"checkpoints", "motion_module.pth")
            
            image_enc = CLIPVisionModelWithProjection.from_pretrained(os.path.join(script_directory,"checkpoints", "image_encoder"))
            image_enc.to(dtype).to(device)
            from diffusers.loaders.single_file_utils import (convert_ldm_vae_checkpoint, convert_ldm_unet_checkpoint, create_text_encoder_from_ldm_clip_checkpoint, create_vae_diffusers_config, create_unet_diffusers_config)
            
            mm.load_model_gpu(model)
            sd = model.model.state_dict_for_saving(None, vae.get_sd(), None)

            # 1. vae
            converted_vae_config = create_vae_diffusers_config(original_config, image_size=512)
            converted_vae = convert_ldm_vae_checkpoint(sd, converted_vae_config)
            vae = AutoencoderKL(**converted_vae_config)
            vae.load_state_dict(converted_vae, strict=False)
            if vae_dtype == "auto":
                try:
                    if mm.should_use_bf16():
                        vae.to(convert_dtype('bf16'))
                    else:
                        vae.to(convert_dtype('fp32'))
                except:
                    raise AttributeError("ComfyUI version too old, can't autodetect properly. Set your dtype manually.")
            else:
                vae.to(convert_dtype(vae_dtype))
            print(f"VAE using dtype: {vae.dtype}")

            # 2. unet
            converted_unet_config = create_unet_diffusers_config(original_config, image_size=512)
            converted_unet = convert_ldm_unet_checkpoint(sd, converted_unet_config)
            del sd
            reference_unet = UNet2DConditionModel(**converted_unet_config)
            reference_unet.load_state_dict(converted_unet, strict=False)

            denoising_unet = UNet3DConditionModel(**ad_unet_config)
            denoising_unet.load_state_dict(converted_unet, strict=False)

            motion_state_dict = torch.load(motion_module_path, map_location="cpu", weights_only=True)
            
            denoising_unet.load_state_dict(motion_state_dict, strict=False)
            del motion_state_dict
            
            guidance_encoder_group = setup_guidance_encoder(cfg)
            
            denoising_unet.load_state_dict(torch.load(denoising_unet_path, map_location="cpu"), strict=False)
            reference_unet.load_state_dict(torch.load(reference_unet_path, map_location="cpu"), strict=False)
           
            denoising_unet.to(dtype).to(device)
            reference_unet.to(dtype).to(device)

            for guidance_type, guidance_encoder_module in guidance_encoder_group.items():
                guidance_encoder_module.load_state_dict(
                    torch.load(
                        osp.join(script_directory,"checkpoints", f"guidance_encoder_{guidance_type}.pth"),
                        map_location="cpu",
                    ),
                    strict=False,
                )
                
            reference_control_writer = ReferenceAttentionControl(
                reference_unet,
                do_classifier_free_guidance=False,
                mode="write",
                fusion_blocks="full",
            )
            reference_control_reader = ReferenceAttentionControl(
                denoising_unet,
                do_classifier_free_guidance=False,
                mode="read",
                fusion_blocks="full",
            )
                
            self.model = ChampModel(
                reference_unet=reference_unet,
                denoising_unet=denoising_unet,
                reference_control_writer=reference_control_writer,
                reference_control_reader=reference_control_reader,
                guidance_encoder_group=guidance_encoder_group,
            ).to(device, dtype=dtype)
            
            if mm.XFORMERS_IS_AVAILABLE:
                reference_unet.enable_xformers_memory_efficient_attention()
                denoising_unet.enable_xformers_memory_efficient_attention()
   
        return (self.model, vae, image_enc)
    
class champ_sampler:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "champ_model": ("CHAMPMODEL",),
            "champ_vae": ("CHAMPVAE",),
            "champ_encoder": ("CHAMPENCODER",),
            "image": ("IMAGE",),
            "width": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 64}),
            "height": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 64}),
            "steps": ("INT", {"default": 20, "min": 1, "max": 200, "step": 1}),
            "guidance_scale": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 20.0, "step": 0.01}),
            "frames": ("INT", {"default": 16, "min": 1, "max": 100, "step": 1}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            "keep_model_loaded": ("BOOLEAN", {"default": True}),
            
            },
    
        }

    RETURN_TYPES = ("IMAGE", "IMAGE",)
    RETURN_NAMES = ("images", "last_image",)
    FUNCTION = "process"
    CATEGORY = "champWrapper"

    def process(self, champ_model, champ_vae, champ_encoder, image, width, height, guidance_scale, steps, seed, keep_model_loaded, frames):
        device = mm.get_torch_device()
        mm.unload_all_models()
        mm.soft_empty_cache()
        model = champ_model
        vae = champ_vae
        image_enc = champ_encoder
        torch.manual_seed(seed)
        dtype = model.reference_unet.dtype
        print(dtype)
        

        config_path = os.path.join(script_directory, "configs/inference.yaml")
        cfg = OmegaConf.load(config_path)

        sched_kwargs = OmegaConf.to_container(cfg.noise_scheduler_kwargs)
        if cfg.enable_zero_snr:
            sched_kwargs.update( 
                rescale_betas_zero_snr=True,
                timestep_spacing="trailing",
                prediction_type="v_prediction",
            )
        noise_scheduler = DDIMScheduler(**sched_kwargs)
        sched_kwargs.update({"beta_schedule": "scaled_linear"})
        
        autocast_condition = (dtype != torch.float32) and not mm.is_device_mps(device)
        with torch.autocast(mm.get_autocast_device(device), dtype=dtype) if autocast_condition else nullcontext():
            image = image.permute(0, 3, 1, 2).to(dtype).to(device)

            B, C, H, W = image.shape
            orig_H, orig_W = H, W
            if W % 64 != 0:
                W = W - (W % 64)
            if H % 64 != 0:
                H = H - (H % 64)
            if orig_H % 64 != 0 or orig_W % 64 != 0:
                image = F.interpolate(image, size=(H, W), mode="bicubic")
           
            B, C, H, W = image.shape
            
            to_pil = transforms.ToPILImage()
            ref_image_pil = to_pil(image[0])
            ref_image_w, ref_image_h = ref_image_pil.size
            
            guidance_pil_group, video_length = combine_guidance_data(cfg, max_files_per_type=frames)
            
            result_video_tensor = inference(
                cfg=cfg,
                vae=vae,
                image_enc=image_enc,
                model=model,
                scheduler=noise_scheduler,
                ref_image_pil=ref_image_pil,
                guidance_pil_group=guidance_pil_group,
                video_length=frames,
                width=width, height=height,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                device=device, dtype=dtype
            )  # (1, c, f, h, w)
            print(result_video_tensor.shape)
            print(result_video_tensor.min(), result_video_tensor.max())
            result_video_tensor = result_video_tensor.squeeze(0)
            result_video_tensor = result_video_tensor.permute(1, 2, 3, 0).cpu()
            print(result_video_tensor.shape)
            return (result_video_tensor,)
def inference(
    cfg,
    vae,
    image_enc,
    model,
    scheduler,
    ref_image_pil,
    guidance_pil_group,
    video_length,
    width,
    height,
    num_inference_steps,
    guidance_scale,
    dtype,
    device
):
    reference_unet = model.reference_unet
    denoising_unet = model.denoising_unet
    guidance_types = cfg.guidance_types
    guidance_encoder_group = {f"guidance_encoder_{g}": getattr(model, f"guidance_encoder_{g}") for g in guidance_types}
    
    generator = torch.Generator(device=device)
    generator.manual_seed(cfg.seed)
    pipeline = MultiGuidance2LongVideoPipeline(
        vae=vae,
        image_encoder=image_enc,
        reference_unet=reference_unet,
        denoising_unet=denoising_unet,
        **guidance_encoder_group,
        scheduler=scheduler,
    )
    pipeline = pipeline.to(device, dtype)
    
    video = pipeline(
        ref_image_pil,
        guidance_pil_group,
        width,
        height,
        video_length,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator
    ).videos
    
    del pipeline
    torch.cuda.empty_cache()
    
    return video       

NODE_CLASS_MAPPINGS = {
    "champ_model_loader": champ_model_loader,
    "champ_sampler": champ_sampler,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "champ_model_loader": "champ_model_loader",
    "champ_sampler": "champ_sampler",
}
