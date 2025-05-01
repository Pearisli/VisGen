import argparse
import math
import os
from copy import deepcopy
from io import BytesIO
from typing import Dict, List, Union

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from PIL import Image
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from omegaconf import OmegaConf

from datasets import load_dataset
from diffusers import AutoencoderKL, DDIMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from src.util.setting import seed_all

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train latent diffusion model with VAE and UNet"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the YAML config file"
    )
    return parser.parse_args()

def setup_directories(base_output: str, job_name: str) -> Dict[str, str]:
    out_run = os.path.join(base_output, job_name)
    dirs = {
        "run": out_run,
        "ckpt": os.path.join(out_run, "checkpoint"),
        "tb": os.path.join(out_run, "tensorboard"),
        "vis": os.path.join(out_run, "visualization"),
    }
    return dirs

@torch.no_grad()
def log_validation(
    vae: AutoencoderKL,
    text_encoder: CLIPTextModel,
    tokenizer: CLIPTokenizer,
    unet: UNet2DConditionModel,
    noise_scheduler: DDIMScheduler,
    valid_dataloader: DataLoader,
    valid_prompt_embeds: Union[List[str], torch.Tensor],
    cfg: OmegaConf,
    accelerator: Accelerator,
    save_directory: str,
    weight_dtype: torch.dtype,
    step: int
):
    pipeline = StableDiffusionPipeline.from_pretrained(
        cfg.pretrained_model_name_or_path,
        vae=accelerator.unwrap_model(vae),
        text_encoder=accelerator.unwrap_model(text_encoder),
        tokenizer=tokenizer,
        unet=accelerator.unwrap_model(unet),
        scheduler=noise_scheduler,
        safety_checker=None,
        torch_dtype=weight_dtype,
        local_files_only=True
    )
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=False)

    generator = torch.Generator(device=accelerator.device)
    if cfg.seed is not None:
        generator = generator.manual_seed(cfg.seed)
    
    with torch.autocast(accelerator.device.type):
        samples = pipeline(
            batch_size=cfg.image_per_row**2,
            generator=generator,
            num_inference_steps=cfg.num_inference_steps,
            prompt_embeds=valid_prompt_embeds,
            output_type="pt"
        )[0]

        valid_loss = 0
        valid_numbers = 0
        progress_bar = tqdm(
            valid_dataloader,
            initial=0,
            desc="Validation Steps",
            disable=not accelerator.is_local_main_process,
            leave=False
        )

        for batch in progress_bar:
            latents = vae.encode(batch["pixel_values"].to(accelerator.device)).latent_dist.sample()
            encoder_hidden_states = text_encoder(batch["input_ids"].to(accelerator.device), return_dict=False)[0]
            latents = latents * vae.config.scaling_factor

            noise = torch.randn_like(latents)

            bsz = latents.shape[0]

            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
            timesteps = timesteps.long()

            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

            model_pred = unet(noisy_latents, timesteps, encoder_hidden_states, return_dict=False)[0]

            loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
            loss = loss.mean(dim=list(range(1, len(loss.shape))))
            valid_loss += loss.detach().sum().item()
            valid_numbers += bsz
    
    accelerator.log({'valid_loss': valid_loss / valid_numbers}, step=step)

    grid = make_grid(samples, nrow=cfg.image_per_row).permute(1, 2, 0).cpu().numpy()
    images = Image.fromarray((grid * 255).astype(np.uint8))
    images.save(os.path.join(save_directory, f"step-{step:06d}.jpg"))

    del pipeline
    torch.cuda.empty_cache()

def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config)

    seed_all(cfg.seed)

    # Create output directories
    job_name = os.path.basename(args.config).split(".")[0]
    dirs = setup_directories(cfg.output_dir, job_name)

    # initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision=cfg.mixed_precision,
        log_with="tensorboard",
        project_config=ProjectConfiguration(project_dir=dirs['run'], logging_dir=dirs['tb'])
    )

    if accelerator.is_main_process:
        for path in dirs.values():
            os.makedirs(path, exist_ok=False)

    # -------------------- Device --------------------
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # -------------------- Model --------------------
    # Load scheduler and models.
    vae: AutoencoderKL = AutoencoderKL.from_pretrained(
        cfg.pretrained_model_name_or_path, subfolder="vae", local_files_only=True
    )
    noise_scheduler: DDIMScheduler = DDIMScheduler.from_pretrained(
        cfg.pretrained_model_name_or_path, subfolder="scheduler", local_files_only=True
    )
    unet = UNet2DConditionModel(
        sample_size=cfg.sample_size,
        in_channels=vae.config.latent_channels,
        out_channels=vae.config.latent_channels,
        block_out_channels=tuple(cfg.block_out_channels),
        attention_head_dim=tuple(cfg.attention_head_dim),
        cross_attention_dim=cfg.cross_attention_dim
    )
    text_encoder = CLIPTextModel.from_pretrained(
        cfg.pretrained_model_name_or_path, subfolder="text_encoder", local_files_only=True
    )
    tokenizer = CLIPTokenizer.from_pretrained(
        cfg.pretrained_model_name_or_path, subfolder="tokenizer", local_files_only=True
    )

    ema_unet = deepcopy(unet)
    ema_unet = EMAModel(
        ema_unet.parameters(),
        decay=0.999,
        model_cls=UNet2DConditionModel,
        model_config=ema_unet.config,
        foreach=True,
    )

    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            ema_unet.save_pretrained(os.path.join(output_dir, "unet_ema"))

            for i, model in enumerate(models):
                model.save_pretrained(os.path.join(output_dir, "unet"))

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()
    
    accelerator.register_save_state_pre_hook(save_model_hook)

    # freeze parameters of models to save more memory
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.enable_xformers_memory_efficient_attention()

    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    if accelerator.is_main_process:
        accelerator.init_trackers("tensorboard")

    # Move unet, vae and text_encoder to device and cast to weight_dtype
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    # -------------------- Optimizer and Learning Rate Scheduler --------------------
    optimizer = optim.AdamW(unet.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    lr_scheduler = get_scheduler(
        name="constant_with_warmup",
        optimizer=optimizer,
        num_warmup_steps=cfg.num_warmup_steps * accelerator.num_processes,
        num_training_steps=cfg.max_train_steps * accelerator.num_processes
    )

    # -------------------- Data --------------------
    # prepare dataset and dataloaders
    dataset = load_dataset(cfg.data_dir)
    train_dataset, valid_dataset = dataset["train"], dataset["validation"]

    def tokenize_captions(examples):
        captions = [caption for caption in examples["text"]]
        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids
    
    # Preprocessing the datasets.  
    train_transforms = transforms.Compose(
        [
            transforms.Resize((cfg.resolution, cfg.resolution)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def preprocess_train(examples):
        images = [Image.open(BytesIO(image)).convert("RGB") for image in examples["image"]]
        examples["pixel_values"] = [train_transforms(image) for image in images]
        examples["input_ids"] = tokenize_captions(examples)
        return examples

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        input_ids = torch.stack([example["input_ids"] for example in examples])
        return {"pixel_values": pixel_values, "input_ids": input_ids}

    with accelerator.main_process_first():
        # Set the training transforms
        train_dataset = train_dataset.with_transform(preprocess_train)
        valid_dataset = valid_dataset.with_transform(preprocess_train)

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=cfg.train_batch_size,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
        # pin_memory=True,
        # persistent_workers=True,
        drop_last=True,
    )

    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=cfg.valid_batch_size,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
        # pin_memory=True,
        # persistent_workers=True,
    )

    valid_prompt = next(iter(valid_dataloader))["input_ids"]
    valid_prompt_embeds = text_encoder(valid_prompt[:cfg.image_per_row**2].to(accelerator.device), return_dict=False)[0]

    # Prepare everything with our `accelerator`.
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )
    ema_unet.to(accelerator.device)

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / accelerator.gradient_accumulation_steps
    )
    max_epoch = math.ceil(cfg.max_train_steps / num_update_steps_per_epoch)

    global_step = 0
    progress_bar = tqdm(
        range(0, cfg.max_train_steps),
        initial=0,
        desc="Training Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(max_epoch):
        unet.train()
        train_loss = 0.0

        for step, batch in enumerate(train_dataloader):

            with accelerator.accumulate(unet):
                # Convert images to latent space
                latents = vae.encode(batch["pixel_values"]).latent_dist.sample()
                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(batch["input_ids"], return_dict=False)[0]
                
                latents = latents * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)

                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
    
                # Predict the noise residual and compute loss
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states, return_dict=False)[0]

                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(cfg.train_batch_size)).mean()
                train_loss += avg_loss.item() / cfg.gradient_accumulation_steps
            
                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), cfg.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                ema_unet.step(unet.parameters())
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if global_step % cfg.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(dirs["ckpt"], f"step-{global_step}")
                        accelerator.save_state(save_path)
                
                if accelerator.is_main_process:
                    if global_step % cfg.validation_steps == 0:
                        ema_unet.store(unet.parameters())
                        ema_unet.copy_to(unet.parameters())
                        log_validation(
                            vae,
                            text_encoder,
                            tokenizer,
                            unet,
                            noise_scheduler,
                            valid_dataloader,
                            valid_prompt_embeds,
                            cfg,
                            accelerator,
                            dirs['vis'],
                            weight_dtype,
                            global_step
                        )
                        ema_unet.restore(unet.parameters())

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= cfg.max_train_steps:
                break

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        ema_unet.copy_to(unet.parameters())

        pipeline = StableDiffusionPipeline.from_pretrained(
            cfg.pretrained_model_name_or_path,
            text_encoder=text_encoder,
            vae=vae,
            unet=unet,
            local_files_only=True
        )
        save_path = os.path.join(dirs["ckpt"], f"step-{global_step}")
        pipeline.save_pretrained(save_path)
        
    accelerator.end_training()

if __name__ == "__main__":
    main()