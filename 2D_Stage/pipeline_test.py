import torch
from tuneavideo.pipelines.pipeline_tuneavideo import TuneAVideoPipeline
from diffusers import AutoencoderKL, DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer, CLIPImageProcessor, CLIPVisionModelWithProjection
from tuneavideo.models.unet_mv2d_condition import UNetMV2DConditionModel
from tuneavideo.models.unet_mv2d_ref import UNetMV2DRefModel
import json
import numpy as np
from PIL import Image
from torchvision import transforms
from einops import rearrange
import os

# Minimal test for pipeline call outside Gradio

def main():
    device = "cpu"
    pretrained_model_path = "stabilityai/stable-diffusion-2-1"
    image_encoder_path = "./models/image_encoder"
    ckpt_dir = "./models/checkpoint"
    validation = {'guidance_scale': 5.0, 'use_inv_latent': False, 'video_length': 4}
    unet_condition_type = "image"
    use_noise = False
    use_shifted_noise = False
    noise_d = 256
    timestep = 40
    val_height = 384
    val_width = 256

    print("Loading models...")
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer", device_map="cpu")
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder", device_map="cpu")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(image_encoder_path, device_map="cpu")
    feature_extractor = CLIPImageProcessor()
    vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
    unet = UNetMV2DConditionModel.from_pretrained_2d(pretrained_model_path, subfolder="unet", local_crossattn=True, device="cpu")
    ref_unet = UNetMV2DRefModel.from_pretrained_2d(pretrained_model_path, subfolder="unet", local_crossattn=True, device="cpu")

    print("Loading checkpoints...")
    unet_params = torch.load(os.path.join(ckpt_dir, "pytorch_model.bin"), map_location="cpu")
    ref_unet_params = torch.load(os.path.join(ckpt_dir, "pytorch_model_1.bin"), map_location="cpu")
    unet.load_state_dict(unet_params)
    ref_unet.load_state_dict(ref_unet_params)

    vae.to(device, dtype=torch.float16)
    unet.to(device, dtype=torch.float16)
    ref_unet.to(device, dtype=torch.float16)
    text_encoder.to(device, dtype=torch.float16)
    image_encoder.to(device, dtype=torch.float16)

    generator = torch.Generator(device=device)
    pipeline = TuneAVideoPipeline(
        vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet, ref_unet=ref_unet,
        feature_extractor=feature_extractor, image_encoder=image_encoder,
        scheduler=DDIMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")
    )
    pipeline.enable_vae_slicing()
    pipeline.set_progress_bar_config(disable=True)

    # Dummy input image (RGBA)
    img = Image.new("RGBA", (val_width, val_height), (128, 128, 128, 255))
    totensor = transforms.ToTensor()
    img_tensor = totensor(np.array(img).astype(np.float32) / 255.)
    imgs_in = rearrange(img_tensor.unsqueeze(0).unsqueeze(0), "B Nv C H W -> (B Nv) C H W")
    imgs_in = imgs_in.to(device, dtype=torch.float16)

    # Dummy pose
    pose_imgs_in = torch.randn(4, 3, val_height, val_width, dtype=torch.float32)
    camera_matrixs = torch.randn(1, 4, 12, dtype=torch.float32)
    prompts = "high quality, best quality"
    prompt_ids = tokenizer(
        prompts, max_length=tokenizer.model_max_length, padding="max_length", truncation=True,
        return_tensors="pt"
    ).input_ids

    print("Calling pipeline...")
    try:
        out = pipeline(
            prompt=prompts,
            image=imgs_in,
            generator=generator,
            num_inference_steps=timestep,
            camera_matrixs=camera_matrixs,
            prompt_ids=prompt_ids,
            height=val_height,
            width=val_width,
            unet_condition_type=unet_condition_type,
            pose_guider=None,
            pose_image=pose_imgs_in,
            use_noise=use_noise,
            use_shifted_noise=use_shifted_noise,
            **validation
        ).videos
        print("Pipeline call succeeded! Output shape:", out.shape)
    except Exception as e:
        import traceback
        print("Pipeline call failed:", e)
        traceback.print_exc()

if __name__ == "__main__":
    main()
