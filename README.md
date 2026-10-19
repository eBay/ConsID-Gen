<div align="center">

# ConsID-Gen
## View-Consistent and Identity-Preserving Image-to-Video Generation

[Mingyang Wu](https://myangwu.github.io/)<sup>1</sup> &nbsp; [Ashirbad Mishra](https://scholar.google.com/citations?user=Cc_213MAAAAJ&hl=en)<sup>2</sup> &nbsp; [Soumik Dey](https://scholar.google.com/citations?user=nlZVZtMAAAAJ&hl=en)<sup>2</sup> &nbsp; [Shuo Xing](https://shuoxing98.github.io/)<sup>1</sup> &nbsp; [Naveen Ravipati](https://www.linkedin.com/in/naveen-ravipati-97983216a/)<sup>2</sup>  
[Hansi Wu](https://scholar.google.co.jp/citations?user=CucpbkEAAAAJ&hl=ja)<sup>2</sup> &nbsp; [Binbin Li](https://scholar.google.com/citations?user=1UlUOMMAAAAJ&hl=en)<sup>2</sup> &nbsp; [Zhengzhong Tu](https://vztu.github.io/)<sup>1</sup><sup>&dagger;</sup>

<sup>1</sup>Texas A&M University &nbsp;&nbsp; <sup>2</sup>eBay Inc.  
<sup>&dagger;</sup>Corresponding Author

### Accepted by CVPR 2026

[![arXiv](https://img.shields.io/badge/arXiv-2602.10113-b31b1b.svg)](https://arxiv.org/abs/2602.10113)
[![Model](https://img.shields.io/badge/Model-HuggingFace-yellow)](https://huggingface.co/mingyang-wu/ConsID-Gen)
[![Code](https://img.shields.io/badge/Code-Internal%20Review-lightgrey)](#open-source-plan)
[![Dataset](https://img.shields.io/badge/Dataset-ConsIDVid-lightgrey)](https://huggingface.co/datasets/mingyang-wu/ConsIDVid)

</div>

---

## Updates

- **2026.03**: Our code and dataset are under internal review.
- **2026.02**: Our paper is accepted by CVPR 2026.


## Open-Source Plan

We will release resources in stages after internal review. Please stay tuned.

- [ ] ConsID-Gen inference/training code
- [ ] ConsIDVid dataset: https://huggingface.co/datasets/mingyang-wu/ConsIDVid
- [ ] Model checkpoints


## Code: ConsID-Gen + Wan2.1 (Inference & Training)

### Setup Environment

```bash
conda create -n considgen python=3.10
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

### Model Download

Use this section to download the model weights required by `run_inference_considgen.py`.

```bash
# 1) Install Hugging Face CLI
pip install "huggingface_hub[cli]"

# 2) Download Wan2.1-Fun-1.3B-InP
mkdir -p ./models/PAI/Wan2.1-Fun-1.3B-InP
hf download "alibaba-pai/Wan2.1-Fun-1.3B-InP" --local-dir "./models/PAI/Wan2.1-Fun-1.3B-InP"

# 3) Download VGGT-1B
mkdir -p ./models/VGGT-1B
hf download "facebook/VGGT-1B" --local-dir "./models/VGGT-1B"

# 4) Download ConsID-Gen
hf download "mingyang-wu/ConsID-Gen" --local-dir "./models/ConsID-Gen/checkpoints"
```

[Google Drive](https://drive.google.com/file/d/1oLiJA7YH71sUrpwquV1IVPo12v4FJfgi/view?usp=drive_link) mirror for ConsID-Gen checkpoints.

### Inference Example

Run single-image conditioned generation with a finetuned checkpoint:

```bash
python run_inference_considgen.py \
  --input_image_path /path/to/input_image.jpg \
  --image_dir /path/to/multi_view_images_dir \
  --prompt "A product-style close-up video with stable lighting and clean background." \
  --output_dir ./tmp \
  --checkpoint_path models/train/ConsID-Gen/model.safetensors
```

### Training

Prepare the training dataset.

The training metadata should be a JSON list. Each sample needs:

- `video`: path to the training video
- `prompt`: text description for the video
- `image_list`: list of multi-view reference image paths

Example metadata structure:

```json
[
  {
    "dataset": "example_dataset_name",
    "video": "/path/to/video.mp4",
    "prompt": "A short text description of the target video.",
    "image_list": [
      "/path/to/view_1.jpg",
      "/path/to/view_2.jpg"
    ]
  }
]
```

```bash
CUDA_VISIBLE_DEVICES=3 python run_train_considgen.py \
  --dataset_metadata_path ./example_metadata.json \
  --model_paths '["models/PAI/Wan2.1-Fun-1.3B-InP/diffusion_pytorch_model.safetensors","models/PAI/Wan2.1-Fun-1.3B-InP/models_t5_umt5-xxl-enc-bf16.pth","models/PAI/Wan2.1-Fun-1.3B-InP/Wan2.1_VAE.pth","models/PAI/Wan2.1-Fun-1.3B-InP/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"]' \
  --vggt_model_path models/VGGT-1B \
  --tokenizer_path models/PAI/Wan2.1-Fun-1.3B-InP/google/umt5-xxl \
  --trainable_models dit,considgen_adapter \
  --output_path models/ConsID-Gen/example \
  --num_epochs 1 \
  --dataset_num_workers 0 \
  --wandb_mode disabled
```

## Citation
If you use the code or any part of the datasets, then please cite this paper in your work.

```bibtex
@InProceedings{Wu_2026_CVPR,
    author    = {Wu, Mingyang and Mishra, Ashirbad and Dey, Soumik and Xing, Shuo and Ravipati, Naveen and Wu, Hansi and Li, Binbin and Tu, Zhengzhong},
    title     = {ConsID-Gen: View-Consistent and Identity-Preserving Image-to-Video Generation},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2026},
    pages     = {1853-1863}
}
```
