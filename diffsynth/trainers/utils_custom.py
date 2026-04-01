import argparse, warnings

import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from tqdm import tqdm

from .utils import DiffusionTrainingModule, ModelLogger, VideoDataset as BaseVideoDataset


class VideoDataset(BaseVideoDataset):
    def __init__(self, metadata_path=None, use_opencv=False, args=None, min_images=None):
        super().__init__(base_path=None, metadata_path=metadata_path, args=args)
        self.use_opencv = use_opencv
        self.min_images = min_images
        self._filter_samples_by_min_images()

    def _filter_samples_by_min_images(self):
        if self.min_images is None or not isinstance(self.data, list):
            return
        filtered_data, removed_count = [], 0
        for sample in tqdm(self.data, desc=f"Filtering samples (min_images={self.min_images})"):
            if "image_list" not in sample or len(sample["image_list"]) >= self.min_images:
                filtered_data.append(sample)
            else:
                removed_count += 1
        self.data = filtered_data
        if removed_count > 0:
            print(f"[INFO] Removed {removed_count} samples with fewer than {self.min_images} images.")

    def load_video(self, file_path):
        if file_path.lower().endswith(".gif"):
            return self._load_gif(file_path)
        return super().load_video(file_path)

    def _build_multi_view_images(self, image_paths):
        images = []
        for image_path in image_paths:
            frames = self.load_data(image_path)
            images.append(frames[0] if isinstance(frames, list) else frames)
        return images

    def __getitem__(self, data_id):
        data = self.data[data_id % len(self.data)].copy()
        for key in ["image", "video"]:
            if key in data:
                path = data[key]
                data[key] = self.load_data(path)
                if data[key] is None:
                    warnings.warn(f"Cannot load file: {path}")
                    return None

        if ("image_list" in self.data_file_keys) and ("image_list" in data):
            data["multi_view_images"] = self._build_multi_view_images(data["image_list"])
        return data


def wan_parser():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--dataset_base_path", type=str, default="", help="Base path of the dataset.")
    parser.add_argument("--dataset_metadata_path", type=str, default="./metadata/metadata_train.json", help="Path to the metadata file of the dataset.")
    parser.add_argument("--max_pixels", type=int, default=1280*720, help="Maximum number of pixels per frame, used for dynamic resolution..")
    parser.add_argument("--height", type=int, default="480", help="Height of images or videos. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--width", type=int, default="832", help="Width of images or videos. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--num_frames", type=int, default=81, help="Number of frames per video. Frames are sampled from the video prefix.")
    parser.add_argument("--data_file_keys", type=str, default="image,video,image_list",help="Data file keys in the metadata. Comma-separated.")
    parser.add_argument("--dataset_repeat", type=int, default=1, help="Number of times to repeat the dataset per epoch.")
    parser.add_argument("--model_paths", type=str, default=None, help="Paths to load models. In JSON format.")
    parser.add_argument("--model_id_with_origin_paths", type=str, default=None, help="Model ID with origin paths, e.g., Wan-AI/Wan2.1-T2V-1.3B:diffusion_pytorch_model*.safetensors. Comma-separated.")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of epochs.")
    parser.add_argument("--output_path", type=str, default="./models/ConsID-Gen", help="Output save path.")
    parser.add_argument("--remove_prefix_in_ckpt", type=str, default="pipe.dit.", help="Remove prefix in ckpt.")
    parser.add_argument("--trainable_models", type=str, default=None, help="Models to train, e.g., dit, vae, text_encoder.")
    parser.add_argument("--lora_base_model", type=str, default=None, help="Which model LoRA is added to.")
    parser.add_argument("--lora_target_modules", type=str, default="q,k,v,o,ffn.0,ffn.2", help="Which layers LoRA is added to.")
    parser.add_argument("--lora_rank", type=int, default=64, help="Rank of LoRA.")
    parser.add_argument("--lora_checkpoint", type=str, default=None, help="Path to the LoRA checkpoint. If provided, LoRA will be loaded from this checkpoint.")
    parser.add_argument("--extra_inputs", default="input_image,multi_view_images",help="Additional model inputs, comma-separated.")
    parser.add_argument("--use_gradient_checkpointing_offload", default=False, action="store_true", help="Whether to offload gradient checkpointing to CPU memory.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument("--max_timestep_boundary", type=float, default=1.0, help="Max timestep boundary (for mixed models, e.g., Wan-AI/Wan2.2-I2V-A14B).")
    parser.add_argument("--min_timestep_boundary", type=float, default=0.0, help="Min timestep boundary (for mixed models, e.g., Wan-AI/Wan2.2-I2V-A14B).")
    parser.add_argument("--find_unused_parameters", default=False, action="store_true", help="Whether to find unused parameters in DDP.")
    parser.add_argument("--save_steps", type=int, default=None, help="Number of checkpoint saving invervals. If None, checkpoints will be saved every epoch.")
    parser.add_argument("--dataset_num_workers", type=int, default=4, help="Number of workers for data loading.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay.")
    parser.add_argument("--vggt_model_path", type=str, default=None, help="Local path to the VGGT model directory.")
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Local path to the tokenizer directory.")
    parser.add_argument("--wandb_project", type=str, default="video_generation", help="Weights & Biases project name.")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="Weights & Biases run name.")
    parser.add_argument("--wandb_entity", type=str, default=None, help="Weights & Biases entity (team/user).")
    parser.add_argument("--wandb_mode", type=str, default=None, help="Weights & Biases mode: online/offline/disabled.")
    return parser


def launch_training_task(
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    num_workers: int = 8,
    save_steps: int = None,
    num_epochs: int = 1,
    gradient_accumulation_steps: int = 1,
    find_unused_parameters: bool = False,
    wandb_project: str = "ebay-videogen-rl",
    wandb_run_name: str = None,
    wandb_entity: str = None,
    wandb_mode: str = None,
):
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, collate_fn=lambda x: x[0], num_workers=num_workers)
    accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation_steps, kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=find_unused_parameters)])
    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)

    wandb_run = None
    if accelerator.is_main_process:
        try:
            import wandb
        except ImportError as e:
            raise ImportError("wandb is required for logging. Please `pip install wandb`.") from e

        wandb_run = wandb.run or wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            entity=wandb_entity,
            mode=wandb_mode,
            config={"num_epochs": num_epochs, "gradient_accumulation_steps": gradient_accumulation_steps, "learning_rate": optimizer.param_groups[0]["lr"]},
        )

    print("[Begin Training]")
    global_step = 0
    for epoch_id in range(num_epochs):
        epoch_loss_sum, epoch_loss_count, accum_loss, accum_count = 0.0, 0, 0.0, 0
        for data in tqdm(dataloader):
            with accelerator.accumulate(model):
                loss = model(data)
                accelerator.backward(loss)
                accum_loss += loss.item(); accum_count += 1
                if not accelerator.sync_gradients:
                    continue

                optimizer.step(); optimizer.zero_grad(); scheduler.step(); global_step += 1
                loss_mean = accelerator.gather(torch.tensor(accum_loss / accum_count, device=loss.device)).mean().item()
                accum_loss, accum_count = 0.0, 0
                epoch_loss_sum += loss_mean; epoch_loss_count += 1

                if accelerator.is_main_process and wandb_run is not None:
                    wandb_run.log({"train/loss_step": loss_mean, "train/lr": scheduler.get_last_lr()[0], "train/epoch": epoch_id + 1, "train/global_step": global_step}, step=global_step)
                model_logger.on_step_end(accelerator, model, save_steps)

        epoch_loss = epoch_loss_sum / max(epoch_loss_count, 1)
        if accelerator.is_main_process:
            print(f"[Epoch {epoch_id + 1}/{num_epochs}] loss={epoch_loss:.6f}")
            if wandb_run is not None:
                wandb_run.log({"train/loss_epoch": epoch_loss, "train/epoch": epoch_id + 1}, step=global_step)

        if save_steps is None:
            model_logger.on_epoch_end(accelerator, model, epoch_id)

    model_logger.on_training_end(accelerator, model, save_steps)
    if accelerator.is_main_process and wandb_run is not None:
        wandb_run.finish()
