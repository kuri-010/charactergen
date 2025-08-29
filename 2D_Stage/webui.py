import gradio as gr
from PIL import Image
import glob

import io
import argparse
import inspect
import os
import random
from typing import Dict, Optional, Tuple
from omegaconf import OmegaConf
import numpy as np

import torch
import torch.utils.checkpoint

from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.utils import check_min_version
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer, CLIPImageProcessor, CLIPVisionModelWithProjection
from torchvision import transforms

from tuneavideo.models.unet_mv2d_condition import UNetMV2DConditionModel
from tuneavideo.models.unet_mv2d_ref import UNetMV2DRefModel
from tuneavideo.models.PoseGuider import PoseGuider
from tuneavideo.pipelines.pipeline_tuneavideo import TuneAVideoPipeline
from tuneavideo.util import shifted_noise
from einops import rearrange
import PIL
from PIL import Image
from torchvision.utils import save_image
import json
import cv2

import onnxruntime as rt
from huggingface_hub.file_download import hf_hub_download
from rm_anime_bg.cli import get_mask, SCALE

from huggingface_hub import hf_hub_download, list_repo_files

repo_id = "zjpshadow/CharacterGen"
all_files = list_repo_files(repo_id, revision="main")


# Force CPU usage
device = "cpu"
data_type_float = torch.float32

for file in all_files:
    if os.path.exists("../" + file):
        continue
    if file.startswith("2D_Stage"):
        hf_hub_download(repo_id, file, local_dir="../")

class rm_bg_api:

    def __init__(self, force_cpu: Optional[bool] = True):
        session_infer_path = hf_hub_download(
            repo_id="skytnt/anime-seg", filename="isnetis.onnx",
        )
        providers: list[str] = ["CPUExecutionProvider"]
        self.session_infer = rt.InferenceSession(
            session_infer_path, providers=providers,
        )

    def remove_background(
        self,
        imgs: list[np.ndarray],
        alpha_min: float,
        alpha_max: float,
    ) -> list:
        process_imgs = []
        for img in imgs:
            # CHANGE to RGB
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            mask = get_mask(self.session_infer, img)

            mask[mask < alpha_min] = 0.0  # type: ignore
            mask[mask > alpha_max] = 1.0  # type: ignore

            img_after = (mask * img + SCALE * (1 - mask)).astype(np.uint8)  # type: ignore
            mask = (mask * SCALE).astype(np.uint8)  # type: ignore
            img_after = np.concatenate([img_after, mask], axis=2, dtype=np.uint8)
            mask = mask.repeat(3, axis=2)
            process_imgs.append(Image.fromarray(img_after))
        return process_imgs

check_min_version("0.24.0")

logger = get_logger(__name__, log_level="INFO")

def set_seed(seed):
    seed = int(seed)  # Ensure seed is always an integer
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def get_bg_color(bg_color):
    if bg_color == 'white':
        bg_color = np.array([1., 1., 1.], dtype=np.float32)
    elif bg_color == 'black':
        bg_color = np.array([0., 0., 0.], dtype=np.float32)
    elif bg_color == 'gray':
        bg_color = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    elif bg_color == 'random':
        bg_color = np.random.rand(3)
    elif isinstance(bg_color, float):
        bg_color = np.array([bg_color] * 3, dtype=np.float32)
    else:
        raise NotImplementedError
    return bg_color

def process_image(image, totensor):
    if not image.mode == "RGBA":
        image = image.convert("RGBA")

    # Find non-transparent pixels
    non_transparent = np.nonzero(np.array(image)[..., 3])
    min_x, max_x = non_transparent[1].min(), non_transparent[1].max()
    min_y, max_y = non_transparent[0].min(), non_transparent[0].max()    
    image = image.crop((min_x, min_y, max_x, max_y))

    # paste to center
    max_dim = max(image.width, image.height)
    max_height = max_dim
    max_width = int(max_dim / 3 * 2)
    new_image = Image.new("RGBA", (max_width, max_height))
    left = (max_width - image.width) // 2
    top = (max_height - image.height) // 2
    new_image.paste(image, (left, top))

    # Reduce image size for debugging and lower memory usage
    image = new_image.resize((256, 384), resample=PIL.Image.BICUBIC)
    image = np.array(image)
    image = image.astype(np.float32) / 255.
    assert image.shape[-1] == 4  # RGBA
    alpha = image[..., 3:4]
    bg_color = get_bg_color("gray")
    image = image[..., :3] * alpha + bg_color * (1 - alpha)
    # save image
    # new_image = Image.fromarray((image * 255).astype(np.uint8))
    # new_image.save("input.png")
    return totensor(image)

class Inference_API:

    def __init__(self):
        self.validation_pipeline = None

    @torch.no_grad()
    def inference(self, input_image, vae, feature_extractor, image_encoder, unet, ref_unet, tokenizer, text_encoder, pretrained_model_path, generator, validation, val_width, val_height, unet_condition_type,
                    pose_guider=None, use_noise=True, use_shifted_noise=False, noise_d=256, crop=False, seed=100, timestep=20):
        try:
            print("inference: set_seed")
            set_seed(seed)
            print("inference: pipeline check")
            # Get the validation pipeline
            if self.validation_pipeline is None:
                print("inference: building pipeline")
                noise_scheduler = DDIMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")
                if use_shifted_noise:
                    print(f"enable shifted noise for {val_height} to {noise_d}")
                    betas = shifted_noise(noise_scheduler.betas, image_d=val_height, noise_d=noise_d)
                    noise_scheduler.betas = betas
                    noise_scheduler.alphas = 1 - betas
                    noise_scheduler.alphas_cumprod = torch.cumprod(noise_scheduler.alphas, dim=0)
                self.validation_pipeline = TuneAVideoPipeline(
                    vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet, ref_unet=ref_unet,feature_extractor=feature_extractor,image_encoder=image_encoder,
                    scheduler=noise_scheduler
                )
                self.validation_pipeline.enable_vae_slicing()
                self.validation_pipeline.set_progress_bar_config(disable=True)

            print("inference: preparing tensors")
            totensor = transforms.ToTensor()

            print("inference: loading pose.json")
            metas = json.load(open("./material/pose.json", "r"))
            cameras = []
            pose_images = []
            input_path = "./material"
            for lm in metas:
                cameras.append(torch.tensor(np.array(lm[0]).reshape(4, 4).transpose(1,0)[:3, :4]).reshape(-1))
                if not crop:
                    pose_images.append(totensor(np.asarray(Image.open(os.path.join(input_path, lm[1])).resize(
                        (val_height, val_width), resample=PIL.Image.BICUBIC)).astype(np.float32) / 255.))
                else:
                    pose_image = Image.open(os.path.join(input_path, lm[1]))
                    crop_area = (128, 0, 640, 768)
                    pose_images.append(totensor(np.array(pose_image.crop(crop_area)).astype(np.float32)) / 255.)
            print("inference: stacking camera matrices and pose images")
            camera_matrixs = torch.stack(cameras).unsqueeze(0).to(device)
            pose_imgs_in = torch.stack(pose_images).to(device)
            # Resize pose_imgs_in to match imgs_in spatial dimensions if needed
            if pose_imgs_in.shape[2:] != imgs_in.shape[2:]:
                import torch.nn.functional as F
                pose_imgs_in = F.interpolate(pose_imgs_in, size=imgs_in.shape[2:], mode='bilinear', align_corners=False)
                print(f"[DEBUG] Resized pose_imgs_in to: {pose_imgs_in.shape}")
            prompts = "high quality, best quality"
            print("inference: tokenizing prompts")
            prompt_ids = tokenizer(
                prompts, max_length=tokenizer.model_max_length, padding="max_length", truncation=True,
                return_tensors="pt"
            ).input_ids[0]

            print("inference: preparing input image")
            # (B*Nv, 3, H, W)
            B = 1
            weight_dtype = data_type_float #7-23-2024 Changed to allow GPU with compute < 8
            imgs_in = process_image(input_image, totensor)
            imgs_in = rearrange(imgs_in.unsqueeze(0).unsqueeze(0), "B Nv C H W -> (B Nv) C H W")
            imgs_in = imgs_in.to(device, dtype=torch.float16)  # Ensure input is float16
            print("inference: running pipeline")
            # Fix types and shapes for pipeline inputs
            val_height_int = int(val_height)
            val_width_int = int(val_width)
            camera_matrixs_fixed = camera_matrixs.to(torch.float32)
            # Ensure prompt_ids has batch dimension if needed
            if prompt_ids.dim() == 1:
                prompt_ids_fixed = prompt_ids.unsqueeze(0)
            else:
                prompt_ids_fixed = prompt_ids
            print(f"  prompts: {prompts} (type: {type(prompts)})")
            print(f"  imgs_in: {imgs_in.shape}, dtype: {imgs_in.dtype}, device: {imgs_in.device}")
            print(f"  generator: {generator}")
            print(f"  timestep: {timestep}")
            print(f"  camera_matrixs: {camera_matrixs_fixed.shape}, dtype: {camera_matrixs_fixed.dtype}, device: {camera_matrixs_fixed.device}")
            print(f"  prompt_ids: {prompt_ids_fixed.shape}, dtype: {prompt_ids_fixed.dtype}")
            print(f"  val_height: {val_height_int}, val_width: {val_width_int}")
            print(f"  unet_condition_type: {unet_condition_type}")
            print(f"  pose_imgs_in: {pose_imgs_in.shape}, dtype: {pose_imgs_in.dtype}, device: {pose_imgs_in.device}")
            print(f"  use_noise: {use_noise}, use_shifted_noise: {use_shifted_noise}")
            print(f"  validation: {validation}")
            try:
                out = self.validation_pipeline(
                    prompt=prompts,
                    image=imgs_in,
                    generator=generator,
                    num_inference_steps=timestep,
                    camera_matrixs=camera_matrixs_fixed,
                    prompt_ids=prompt_ids_fixed,
                    height=val_height_int,
                    width=val_width_int,
                    unet_condition_type=unet_condition_type,
                    pose_guider=None,
                    pose_image=pose_imgs_in,
                    use_noise=use_noise,
                    use_shifted_noise=use_shifted_noise,
                    **validation
                ).videos
                print("inference: pipeline finished")
            except Exception as e:
                import traceback
                print("Error during pipeline call:", e)
                traceback.print_exc()
                return [None, None, None, None]
            out = rearrange(out, "B C f H W -> (B f) C H W", f=validation.video_length)

            image_outputs = []
            print("inference: saving images")
            for bs in range(4):
                img_buf = io.BytesIO()
                save_image(out[bs], img_buf, format='PNG')
                img_buf.seek(0)
                img = Image.open(img_buf)
                image_outputs.append(img)
            print("inference: done")
            # torch.cuda.empty_cache()  # Not needed for CPU
            return image_outputs 
        except Exception as e:
            import traceback
            print("Error in inference:", e)
            traceback.print_exc()
            return [None, None, None, None]

@torch.no_grad()
def main(
    pretrained_model_path: str,
    image_encoder_path: str,
    ckpt_dir: str,
    validation: Dict,
    local_crossattn: bool = True,
    unet_from_pretrained_kwargs=None,
    unet_condition_type=None,
    use_pose_guider=False,
    use_noise=True,
    use_shifted_noise=False,
    noise_d=256
):
    *_, config = inspect.getargvalues(inspect.currentframe())

    device = "cpu"

    import psutil
    print(f"[DEBUG] Available RAM before loading models: {psutil.virtual_memory().available / (1024**3):.2f} GB")
    print(f"[DEBUG] Loading tokenizer from: {pretrained_model_path}/tokenizer")
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer", device_map="cpu")
    print(f"[DEBUG] Loading text_encoder from: {pretrained_model_path}/text_encoder")
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder", device_map="cpu")
    print(f"[DEBUG] Loading image_encoder from: {image_encoder_path}")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(image_encoder_path, device_map="cpu")
    feature_extractor = CLIPImageProcessor()
    print(f"[DEBUG] Loading VAE from: {pretrained_model_path}/vae")
    vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
    print(f"[DEBUG] Loading UNet from: {pretrained_model_path}/unet")
    unet = UNetMV2DConditionModel.from_pretrained_2d(pretrained_model_path, subfolder="unet", local_crossattn=local_crossattn, device="cpu", **unet_from_pretrained_kwargs)
    print(f"[DEBUG] Loading RefUNet from: {pretrained_model_path}/unet")
    ref_unet = UNetMV2DRefModel.from_pretrained_2d(pretrained_model_path, subfolder="unet", local_crossattn=local_crossattn, device="cpu", **unet_from_pretrained_kwargs)
    print(f"[DEBUG] Available RAM after loading models: {psutil.virtual_memory().available / (1024**3):.2f} GB")
    if use_pose_guider:
        pose_guider = PoseGuider(noise_latent_channels=4).to(device)
    else:
        pose_guider = None

    print(f"[DEBUG] Loading UNet params from: {os.path.join(ckpt_dir, 'pytorch_model.bin')}")
    unet_params = torch.load(os.path.join(ckpt_dir, "pytorch_model.bin"), map_location="cpu")
    print(f"[DEBUG] UNet params keys: {list(unet_params.keys())[:5]} ... total: {len(unet_params)}")
    if use_pose_guider:
        print(f"[DEBUG] Loading pose_guider params from: {os.path.join(ckpt_dir, 'pytorch_model_1.bin')}")
        pose_guider_params = torch.load(os.path.join(ckpt_dir, "pytorch_model_1.bin"), map_location="cpu")
        print(f"[DEBUG] Loading ref_unet params from: {os.path.join(ckpt_dir, 'pytorch_model_2.bin')}")
        ref_unet_params = torch.load(os.path.join(ckpt_dir, "pytorch_model_2.bin"), map_location="cpu")
        pose_guider.load_state_dict(pose_guider_params)
    else:
        print(f"[DEBUG] Loading ref_unet params from: {os.path.join(ckpt_dir, 'pytorch_model_1.bin')}")
        ref_unet_params = torch.load(os.path.join(ckpt_dir, "pytorch_model_1.bin"), map_location="cpu")
    print(f"[DEBUG] Loading state dicts into models...")
    unet.load_state_dict(unet_params)
    ref_unet.load_state_dict(ref_unet_params)
    print(f"[DEBUG] Model state dicts loaded.")

    weight_dtype = torch.float16

    text_encoder.to(device, dtype=weight_dtype)
    image_encoder.to(device, dtype=weight_dtype)
    vae.to(device, dtype=weight_dtype)
    ref_unet.to(device, dtype=weight_dtype)
    unet.to(device, dtype=weight_dtype)

    vae.requires_grad_(False)
    unet.requires_grad_(False)
    ref_unet.requires_grad_(False)

    generator = torch.Generator(device=device)
    inferapi = Inference_API()
    remove_api = rm_bg_api()
    def gen4views(image, width, height, seed, timestep, remove_bg):
        print("gen4views called")
        try:
            print("Before remove_bg")
            if remove_bg:
                image = remove_api.remove_background(
                    imgs=[np.array(image)],
                    alpha_min=0.1,
                    alpha_max=0.9,
                )[0]
                print("After remove_bg")
            print("Before inference")
            result = inferapi.inference(
                image, vae, feature_extractor, image_encoder, unet, ref_unet, tokenizer, text_encoder, pretrained_model_path,
                generator, validation, width, height, unet_condition_type,
                pose_guider=pose_guider, use_noise=use_noise, use_shifted_noise=use_shifted_noise, noise_d=noise_d,
                crop=True, seed=seed, timestep=timestep
            )
            print("After inference")
            return result
        except Exception as e:
            import traceback
            print("Error in gen4views:", e)
            traceback.print_exc()
            return [None, None, None, None]

    with gr.Blocks() as demo:
        gr.Markdown("# [SIGGRAPH'24] CharacterGen: Efficient 3D Character Generation from Single Images with Multi-View Pose Calibration")
        gr.Markdown("# 2D Stage: One Image to Four Views of Character Image")
        gr.Markdown("**Please Upload the Image without background, and the pictures uploaded should preferably be full-body frontal photos.**")
        with gr.Row():
            with gr.Column():
                img_input = gr.Image(type="pil", label="Upload Image(without background)", image_mode="RGBA", width=768, height=512)
                gr.Examples(
                    label="Example Images",
                    examples=glob.glob("./material/examples/*.png"),
                    inputs=[img_input]
                )
                with gr.Row():
                    width_input = gr.Number(label="Width", value=512)
                    height_input = gr.Number(label="Height", value=768)
                    seed_input = gr.Number(label="Seed", value=2333)
                    remove_bg = gr.Checkbox(label="Remove Background (with algorithm)", value=False)
                timestep = gr.Slider(minimum=10, maximum=70, step=1, value=40, label="Timesteps")
            with gr.Column():
                button = gr.Button(value="Generate")
                output = gr.Gallery(label="4 views of Character Image")
        
        button.click(
            fn=gen4views,
            inputs=[img_input, width_input, height_input, seed_input, timestep, remove_bg],
            outputs=[output]
        )

    demo.launch(share=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/infer.yaml")
    args = parser.parse_args()

    import sys
    import traceback
    def handle_exception(exc_type, exc_value, exc_traceback):
        print("Uncaught exception:", exc_value)
        traceback.print_exception(exc_type, exc_value, exc_traceback)
    sys.excepthook = handle_exception
    main(**OmegaConf.load(args.config))
