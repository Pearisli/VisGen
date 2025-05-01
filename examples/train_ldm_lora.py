import argparse
import math
import os
from io import BytesIO
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from PIL import Image
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

import tensorboard
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from omegaconf import OmegaConf
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict

from datasets import load_dataset
from diffusers.training_utils import cast_training_params
from diffusers.utils import convert_state_dict_to_diffusers
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler, StableDiffusionPipeline
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
    pipeline: StableDiffusionPipeline,
    validation_prompts: List[str],
    cfg: OmegaConf,
    accelerator: Accelerator,
    save_directory: str,
    step: int
):
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=False)
    generator = torch.Generator(device=accelerator.device)
    if cfg.seed is not None:
        generator = generator.manual_seed(cfg.seed)
    
    with torch.autocast(accelerator.device.type):
        samples = pipeline(
            validation_prompts,
            height=cfg.resolution,
            width=cfg.resolution,
            num_inference_steps=cfg.num_inference_steps,
            generator=generator,
            output_type="pt"
        ).images

    grid = make_grid(samples, nrow=cfg.nrow).permute(1, 2, 0).cpu().numpy()
    images = Image.fromarray((grid * 255).astype(np.uint8))
    images.save(os.path.join(save_directory, f"step-{step:06d}.jpg"))

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
    # Load scheduler, tokenizer and models.
    tokenizer = CLIPTokenizer.from_pretrained(
        cfg.pretrained_model_name_or_path, subfolder="tokenizer", local_files_only=True
    )
    text_encoder = CLIPTextModel.from_pretrained(
        cfg.pretrained_model_name_or_path, subfolder="text_encoder", local_files_only=True
    )
    vae: AutoencoderKL = AutoencoderKL.from_pretrained(
        cfg.pretrained_model_name_or_path, subfolder="vae", local_files_only=True
    )
    unet: UNet2DConditionModel = UNet2DConditionModel.from_pretrained(
        cfg.pretrained_model_name_or_path, subfolder="unet", local_files_only=True
    )
    noise_scheduler: DDIMScheduler = DDIMScheduler.from_pretrained(
        cfg.pretrained_model_name_or_path, subfolder="scheduler", local_files_only=True
    )

    # freeze parameters of models to save more memory
    unet.requires_grad_(False)
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

    unet_lora_config = LoraConfig(
        r=cfg.rank,
        lora_alpha=cfg.rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    unet.add_adapter(unet_lora_config)

    # Move unet, vae and text_encoder to device and cast to weight_dtype
    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    if cfg.mixed_precision == "fp16":
        # only upcast trainable parameters (LoRA) into fp32
        cast_training_params(unet, dtype=torch.float32)

    lora_layers = filter(lambda p: p.requires_grad, unet.parameters())

    # -------------------- Optimizer and Learning Rate Scheduler --------------------
    optimizer = optim.AdamW(lora_layers, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

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
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def unwrap_model(model):
        return accelerator.unwrap_model(model)
    
    def preprocess_train(examples):
        images = [Image.open(BytesIO(image)).convert("RGB") for image in examples["image"]]
        examples["pixel_values"] = [train_transforms(image) for image in images]
        examples["input_ids"] = tokenize_captions(examples)
        return examples

    with accelerator.main_process_first():
        # Set the training transforms
        train_dataset = train_dataset.with_transform(preprocess_train)

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        input_ids = torch.stack([example["input_ids"] for example in examples])
        return {"pixel_values": pixel_values, "input_ids": input_ids}

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=cfg.train_batch_size,
        collate_fn=collate_fn,
        num_workers=16,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True,
    )

    validation_prompts = valid_dataset[:cfg.nrow ** 2]["text"]

    # Prepare everything with our `accelerator`.
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

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
                latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
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

                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(batch["input_ids"], return_dict=False)[0]

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
                    params_to_clip = lora_layers
                    accelerator.clip_grad_norm_(params_to_clip, cfg.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if global_step % cfg.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(dirs["ckpt"], f"step-{global_step}")
                        unwrapped_unet = unwrap_model(unet)

                        unet_lora_state_dict = convert_state_dict_to_diffusers(
                            get_peft_model_state_dict(unwrapped_unet)
                        )

                        StableDiffusionPipeline.save_lora_weights(
                            save_directory=save_path,
                            unet_lora_layers=unet_lora_state_dict,
                            safe_serialization=True,
                        )
                
                if accelerator.is_main_process:
                    if global_step % cfg.validation_steps == 0:
                        pipeline = StableDiffusionPipeline.from_pretrained(
                            cfg.pretrained_model_name_or_path,
                            unet=unwrap_model(unet),
                            torch_dtype=weight_dtype,
                        )

                        log_validation(
                            pipeline=pipeline,
                            validation_prompts=validation_prompts,
                            accelerator=accelerator,
                            cfg=cfg,
                            save_directory=dirs['vis'],
                            step=global_step
                        )

                        del pipeline
                        torch.cuda.empty_cache()

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= cfg.max_train_steps:
                break

    # Save the lora layers
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = unet.to(torch.float32)

        save_path = os.path.join(dirs["ckpt"], f"step-{global_step}")
        unwrapped_unet = unwrap_model(unet)
        unet_lora_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(unwrapped_unet))
        StableDiffusionPipeline.save_lora_weights(
            save_directory=save_path,
            unet_lora_layers=unet_lora_state_dict,
            safe_serialization=True,
        )
    
    accelerator.end_training()

if __name__ == "__main__":
    main()
