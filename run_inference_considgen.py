import argparse
import os
from glob import glob

import imageio
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from diffsynth.pipelines.wan_video_considgen import WanVideoConsIDGenPipeline, ModelConfig


def _resolve_required_paths(model_dir: str, pattern: str):
    paths = sorted(glob(os.path.join(model_dir, pattern)))
    if not paths:
        raise FileNotFoundError(f"Missing required weights: pattern='{pattern}' under '{model_dir}'")
    return paths[0] if len(paths) == 1 else paths


def save_video_ref_input(frames, save_path, fps, quality=9, ffmpeg_params=None, ref_size=None):
    writer = imageio.get_writer(save_path, fps=fps, quality=quality, ffmpeg_params=ffmpeg_params)
    if ref_size is None:
        ref_size = frames[0].size if isinstance(frames[0], Image.Image) else (frames[0].shape[1], frames[0].shape[0])

    for frame in tqdm(frames, desc="Saving video"):
        if not isinstance(frame, Image.Image):
            frame = Image.fromarray(np.uint8(frame))
        frame = frame.resize(ref_size, Image.BICUBIC)
        writer.append_data(np.array(frame))
    writer.close()


def build_ConsID_Gen_pipeline(
    checkpoint_path: str = None,
    use_usp: bool = False,
    wan_model_dir: str = "models/PAI/Wan2.1-Fun-1.3B-InP",
    vggt_model_path: str = None,
    tokenizer_path: str = None,
):
    model_configs = [
        ModelConfig(path=_resolve_required_paths(wan_model_dir, "diffusion_pytorch_model*.safetensors"), offload_device="cpu"),
        ModelConfig(path=_resolve_required_paths(wan_model_dir, "models_t5_umt5-xxl-enc-bf16.pth"), offload_device="cpu"),
        ModelConfig(path=_resolve_required_paths(wan_model_dir, "Wan2.1_VAE.pth"), offload_device="cpu"),
        ModelConfig(path=_resolve_required_paths(wan_model_dir, "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"), offload_device="cpu"),
    ]
    pipe = WanVideoConsIDGenPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        use_usp=use_usp,
        redirect_common_files=False,
        vggt_model_path=vggt_model_path,
        tokenizer_path=tokenizer_path,
        model_configs=model_configs,
    )

    if checkpoint_path:
        pipe.load_checkpoint_and_modules(
            checkpoint_path=checkpoint_path,
            module_mappings={"pipe.considgen_adapter.": pipe.considgen_adapter},
        )
        print(f"[INFO] Loaded checkpoint: {checkpoint_path}")

    pipe.enable_vram_management()
    return pipe


def load_multi_view_images(image_dir: str):
    image_paths = sorted(glob(os.path.join(image_dir, "*")))
    valid_ext = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    image_paths = [p for p in image_paths if os.path.splitext(p)[1].lower() in valid_ext]
    if not image_paths:
        raise ValueError(f"No valid images found in {image_dir}")
    return [Image.open(p).convert("RGB") for p in image_paths]


def run_single_inference(args):
    os.makedirs(args.output_dir, exist_ok=True)

    first_frame = Image.open(args.input_image_path).convert("RGB")
    multi_view_images = load_multi_view_images(args.image_dir)

    pipe = build_ConsID_Gen_pipeline(
        checkpoint_path=args.checkpoint_path,
        use_usp=args.use_usp,
        wan_model_dir=args.wan_model_dir,
        vggt_model_path=args.vggt_model_path,
        tokenizer_path=args.tokenizer_path,
    )

    result = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        input_image=first_frame,
        multi_view_images=multi_view_images,
        seed=args.seed,
        tiled=args.tiled,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
    )

    image_name = os.path.splitext(os.path.basename(args.input_image_path))[0]
    output_path = os.path.join(args.output_dir, f"{image_name}_custom_videogen.mp4")
    ref_size = first_frame.size if isinstance(first_frame, Image.Image) else (first_frame.shape[1], first_frame.shape[0])

    save_video_ref_input(result, output_path, fps=args.fps, quality=args.quality, ref_size=ref_size)
    print(f"[INFO] Saved video to: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Single inference for custom CondID-Gen pipeline")
    parser.add_argument("--input_image_path", type=str, required=True, help="Input image path used as input_image.")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory of multi-view reference images.")
    parser.add_argument("--prompt", type=str, required=True, help="Positive prompt.")
    parser.add_argument("--negative_prompt", type=str, default="low quality, blurry, distorted, artifact, watermark, text")
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--checkpoint_path", type=str, default="models/ConsID-Gen/checkpoints/model.safetensors", help="Path to full finetune checkpoint.")
    parser.add_argument("--use_usp", action="store_true", default=False)
    parser.add_argument("--wan_model_dir", type=str, default="models/PAI/Wan2.1-Fun-1.3B-InP")

    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--quality", type=int, default=5)

    parser.add_argument("--tiled", dest="tiled", action="store_true")
    parser.add_argument("--no_tiled", dest="tiled", action="store_false")
    parser.set_defaults(tiled=True)

    parser.add_argument("--vggt_model_path", type=str, default="models/VGGT-1B", help="Local path to the VGGT model directory.")
    parser.add_argument("--tokenizer_path", type=str, default="models/PAI/Wan2.1-Fun-1.3B-InP/google/umt5-xxl", help="Local path to the tokenizer directory.")
    return parser.parse_args()


if __name__ == "__main__":
    run_single_inference(parse_args())
