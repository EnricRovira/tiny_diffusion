import logging
import os
import random
from functools import partial
import signal
from datetime import datetime
import click
import numpy as np
import torch
import torch.optim as optim
import wandb
from transformers import get_cosine_schedule_with_warmup
from diffusers.image_processor import VaeImageProcessor
from torchvision.utils import make_grid 
from PIL import Image

from model.adopt import ADOPT
# from model.dit import DiT
from model.mmdit import MMDiT
from model.rectified_flow import RectifiedFlow
from model.dit_utils import build_dataloader, load_encoders
from data.prepare_data import encode_prompt_with_t5

# Enable TF32 for faster training
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.unique_kernel_names = True
torch._inductor.config.fx_graph_cache = True

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WDS_VERBOSE_CACHE"] = "1"
os.environ["GOPEN_VERBOSE"] = "0"
torch.cuda.empty_cache()

CAPTURE_INPUT = False


def generate_image(
    diffusion_model,
    vae_model,
    text_encoder,
    tokenizer,
    device,
    prompt: str,
    neg_prompt: str = None,
    seed: int = 12,
    cfg: float = 4.,
    num_steps: int = 50
):
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)
    else:
        generator = None
        
    in_channels, latent_size = 16, 32
    conds = encode_prompt_with_t5(
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        prompt=[prompt],
        device=device,
        max_sequence_length=128
    )
    if neg_prompt:
        null_cond= encode_prompt_with_t5(
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            prompt=[""],
            device=device,
            max_sequence_length=128
        )
    else:
        null_cond = None
    init_noise = torch.randn(
        size=(1, in_channels, latent_size, latent_size),
        device=device, dtype=torch.bfloat16, generator=generator
    ) 
    with torch.amp.autocast(device_type=device, dtype=torch.bfloat16):
        images = diffusion_model.sample_with_xps(
            init_noise, conds, null_cond=null_cond, sample_steps=num_steps, cfg=cfg
        )

    imgs_decoded = [
        vae_model.decode(
            (img.to(torch.float32) / vae_model.config.scaling_factor) + vae_model.config.shift_factor
        ).sample.cpu().squeeze(0)
        for img in images
    ]
    grid = make_grid(imgs_decoded, nrow=8)
    grid_image = Image.fromarray(
        (np.clip(grid.numpy().transpose(1, 2, 0) * 0.5 + 0.5, 0, 1) * 255).astype(np.uint8)
    ).resize((1024, 1024))
    return grid_image


def evaluate_samples(
    wandb,
    global_step,
    diffusion_model,
    vae_model,
    text_encoder,
    tokenizer,
    device,
    seed=12
):
    prompts = [
        "A photo of a cat",
        'The image shows a painting of a tall ship in the water, with its sails billowing in the wind. The ship is surrounded by a vast expanse of blue water, and the sky is a deep blue with a few white clouds. The sun is shining brightly, reflecting off the surface of the water and creating a beautiful contrast between the dark blue of the ship and the bright white of the sky.',
        'The image shows the ruins of an old building in the middle of a field, surrounded by grass, plants, and trees. The sky is visible in the background, and the building appears to be in a state of disrepair, with crumbling walls and broken windows.',
        'The image shows a seagull perched atop a cement pillar in front of the Colosseum in Rome, Italy. The sky is filled with clouds and the buildings of the city can be seen in the background.',
        'The image shows a painting of a river with a house in the background, surrounded by trees, plants, and flowers. The sky is filled with clouds, creating a peaceful atmosphere.',
        'The image shows a small house nestled in the middle of a grassy field surrounded by trees and rocks, with a mountain in the background and clouds in the sky.',
        'The image shows an old black and white photo of a man and woman standing next to each other. The man is wearing a cap and has a cigarette in his mouth, while the woman is wearing spectacles. In the background, there is a wall.',
        'The image shows an old black and white photo of a young boy standing on a bench with a wall in the background. He is wearing a dress and footwear, giving the photo a classic and timeless feel.',
        'The image shows a brown and cream colored frog sitting on top of a green leaf with a blurred background.',
        'The image shows two men walking down a street lined with trees. They are both wearing coats and hats, and one of them is holding a bag. In the background, there is a building with windows, railings, and balconies, and the sky is visible above them.',
        'The image shows a close up of a tree branch with green leaves against a blurred background. The leaves are a vibrant green color, and the background is slightly out of focus, giving the image a dreamy feel.'
    ]
    diffusion_model.model = diffusion_model.model.eval()
    list_grids = []
    for idx, prompt in enumerate(prompts):
        grid_image = generate_image(
            diffusion_model,
            vae_model,
            text_encoder,
            tokenizer,
            device,
            prompt=prompt,
            neg_prompt=None,
            seed=12,
            cfg=4.,
            num_steps=50
        )
        list_grids.append(wandb.Image(grid_image, caption=prompt)) 
    wandb.log({
        "grid_sample": list_grids
    }, step=global_step)

def forward(
    diffusion_model: RectifiedFlow,
    batch,
    vae_model,
    device,
    global_step,
    ctx,
    generator=None,
    binnings=None,
    prob_unconditional=0.1,
):

    (vae_latent, caption_encoded) = batch

    # Make the 5% unconditional
    mask = (torch.rand(caption_encoded.shape[0], 1, 1) <= prob_unconditional).to(device)
    caption_encoded = caption_encoded * mask
    caption_encoded[caption_encoded == 0] = 0

    # assert vae_latent.device == device, f"Device: {vae_latent.device}, expected: {device}"
    # assert vae_latent.dtype == torch.float32, f"Dtype: {vae_latent.dtype}, expected: torch.float32"
    # assert caption_encoded.device == device, f"Device: {caption_encoded.device}, expected: {device}"
    assert caption_encoded.dtype in {torch.float16, torch.bfloat16}, f"Dtype: {caption_encoded.dtype}, expected: torch.float16 or torch.bfloat16"
    batch_size = vae_latent.size(0)

    # Normalize Vae latent
    vae_latent = (vae_latent - vae_model.config.shift_factor) * vae_model.config.scaling_factor
    vae_latent = vae_latent.to(torch.bfloat16)

    # log normal sample
    # z = torch.randn(
    #     batch_size, device=device, dtype=torch.bfloat16, generator=generator
    # )
    # t = torch.nn.Sigmoid()(z)

    if CAPTURE_INPUT and global_step == 0:
        torch.save(vae_latent, f"test_data/vae_latent_{global_step}.pt")
        torch.save(caption_encoded, f"test_data/caption_encoded_{global_step}.pt")
        # torch.save(t, f"test_data/timesteps_{global_step}.pt")

    # noise = torch.randn(
    #     vae_latent.shape, device=device, dtype=torch.bfloat16, generator=generator
    # )
    with ctx:
        # Forward pass
        # tr = t.reshape(batch_size, 1, 1, 1)
        # z_t = vae_latent * (1 - tr) + noise * tr
        # v_objective = vae_latent - noise
        # output = dit_model(z_t, caption_encoded, t)
        loss, outputs = diffusion_model(vae_latent, caption_encoded)
        # print(outputs['batchwise_loss'])
        

        # diffusion_loss_batchwise = (
        #     (v_objective.float() - output.float()).pow(2).mean(dim=(1, 2, 3))
        # )
        # total_loss = diffusion_loss_batchwise.mean()

        # timestep binning
        # if binnings is not None:
        #     (
        #         diffusion_loss_binning,
        #         diffusion_loss_binning_count,
        #     ) = binnings
        #     for element in outputs["batchwise_loss"]:
        #         tv, tloss = element
        #         tv = torch.nn.Sigmoid()(tv).cpu().item()
        #         diffusion_loss_binning[tv] += tloss
        #         diffusion_loss_binning_count[tv] += 1
        # raise ValueError('STOP!!')
    return loss

def save_weights(logger, path_checkpoints, run_name, global_step, diffusion_model):
    logger.info(
        f"Saving checkpoint to {path_checkpoints}/{run_name}/{global_step}.pt"
    )
    os.makedirs(path_checkpoints, exist_ok=True)
    os.makedirs(f"{path_checkpoints}/{run_name}", exist_ok=True)
    torch.save(
        diffusion_model.model.state_dict(),
        f"{path_checkpoints}/{run_name}/{global_step}.pt",
    )


@click.command()
@click.option(
    "--dataset_url",
    type=str,
    default="/mnt/d/pd3m/dataset/{000000..000112}.tar",
    help="URL for training dataset",
)
@click.option(
    "--test_dataset_url",
    type=str,
    default="/mnt/d/pd3m/dataset/{000113..000113}.tar",
    help="URL for test dataset",
)
@click.option("--num_epochs", type=int, default=50, help="Number of training epochs")
@click.option("--batch_size", type=int, default=64, help="Batch size for training")
@click.option("--learning_rate", type=float, default=1e-4, help="Learning rate")
@click.option("--max_steps", type=int, default=200_000, help="Maximum training steps")
@click.option("--log_wandb", type=bool, default=False, help="Log to wandb")
@click.option(
    "--log_every", type=int, default=100, help="Steps between logging"
)
@click.option(
    "--evaluate_every", type=int, default=2_500, help="Steps between evaluations"
)
@click.option(
    "--save_checkpoint_every", type=int, default=10_000, help="Steps between saving checkpoints"
)
@click.option(
    "--run_name", type=str, default=None, help="Name of run"
)
@click.option(
    "--model_width", type=int, default=768, help="Width of the model"
)
@click.option(
    "--model_depth", type=int, default=16, help="Depth of the model"
)
@click.option(
    "--num_heads", type=int, default=12, help="number of heads of the model"
)
@click.option(
    "--learn_sigma", type=bool, default=False, help="Learn sigma"
)
@click.option(
    "--prob_unconditional", type=float, default=0.1, help="Probability of unconditional samples"
)
@click.option(
    "--compile_models", type=bool, default=False, help="Compile models"
)
@click.option(
    "--path_checkpoints", type=str, default="/mnt/d/tiny_diffusion/checkpoints", help="Path to save checkpoints"
)
def train(
    dataset_url,
    test_dataset_url,
    num_epochs,
    batch_size,
    learning_rate,
    max_steps,
    log_wandb,
    log_every,
    evaluate_every,
    save_checkpoint_every,
    run_name,
    model_width,
    model_depth,
    num_heads,
    learn_sigma,
    prob_unconditional,
    compile_models,
    path_checkpoints,
):
    # Initialize distributed training
    assert torch.cuda.is_available(), "CUDA is required for training"
    device = 'cuda'

    # Set random seeds
    torch.manual_seed(12)
    torch.cuda.manual_seed(12)
    np.random.seed(12)
    random.seed(12)

    # Initialize models
    vae_model, tokenizer, text_encoder = load_encoders(
        device=device, compile_models=compile_models,
        vae_path="black-forest-labs/FLUX.1-dev",
        text_encoder_path="EleutherAI/pile-t5-large"
    )

    patch_size = 2
    in_channels = 16
    cross_attn_input_size = 1024
    diffusion_model = RectifiedFlow(
        MMDiT(
            in_channels=in_channels,
            patch_size=patch_size,
            depth=model_depth,
            num_heads=num_heads,
            mlp_ratio=4.0,
            context_input_size=cross_attn_input_size,
            hidden_size=model_width,
            # repa_target_layers=[6],
            # repa_target_dim=1536,
        ).to(device), 
        learn_sigma=learn_sigma
    )

    param_count = sum(p.numel() for p in diffusion_model.model.parameters())
    print(f"Model parameters: {param_count / 1e6 :.4f} M")
    if run_name is None:
        run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    if log_wandb:
        wandb.init(
            project="tiny_diffusion",
            name=run_name,
            config={
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "num_epochs": num_epochs,
                "model_parameters": param_count / 1e6,
                "model_width": model_width,
                "model_depth": model_depth,
                "num_heads": num_heads,
                "patch_size": patch_size,
                "in_channels": in_channels,
                "cross_attn_input_size": cross_attn_input_size,
            },
        )


    if compile_models:
        diffusion_model.model = torch.compile(diffusion_model.model, backend='inductor', dynamic=True)

    optimizer = optim.AdamW(
        params=diffusion_model.model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.95),
        fused=True,
    )

    num_warmup_steps = max_steps // 20
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps, max_steps
    )

    # Create dataloaders
    train_loader = build_dataloader(
        base_path=dataset_url, batch_size=batch_size, num_workers=4, shuffle=True,
        pin_memory=True, max_items_per_epoch=1_100_000, persistent_workers=True
    )
    test_loader = build_dataloader(
        base_path=test_dataset_url, batch_size=batch_size, num_workers=1, shuffle=False,
        pin_memory=True, max_items_per_epoch=2_000
    )

    # Setup automatic mixed precision
    ctx = torch.amp.autocast(device_type=device, dtype=torch.bfloat16)

    # Setup logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Initialize step counter
    global_step = 0

    # Training loop
    diffusion_model.model.train()

    diffusion_loss_binning = {k: 0 for k in range(10)}
    diffusion_loss_binning_count = {k: 0 for k in range(10)}

    logger.info(f"=> Starting training for {num_epochs} epochs and {max_steps} steps")
    for epoch in range(num_epochs):
        if global_step >= max_steps:
            break

        for train_batch_idx, train_batch in enumerate(train_loader):
            if global_step >= max_steps:
                break
            
            train_batch = (
                train_batch[0].to(device, non_blocking=True), 
                train_batch[1].to(device, non_blocking=True)
            )
            total_loss = forward(
                diffusion_model=diffusion_model,
                batch=train_batch,
                vae_model=vae_model,
                device=device,
                global_step=global_step,
                ctx=ctx,
                binnings=(
                    diffusion_loss_binning,
                    diffusion_loss_binning_count,
                ),
                prob_unconditional=prob_unconditional,
            )
            # Optimization step
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            lr_scheduler.step()

            # Logging
            if global_step % log_every == 0:
                total_loss = total_loss.item()

                if log_wandb:
                    wandb.log(
                        {
                            "Datapoints": global_step * batch_size,
                            "train/total_loss": total_loss,
                            "train/learning_rate": lr_scheduler.get_last_lr()[0],
                            "train/epoch": epoch,
                            "train/step": global_step,
                            "train_binning/diffusion_loss_binning": {
                                k: v / max(c, 1)
                                for k, v, c in zip(
                                    diffusion_loss_binning.keys(),
                                    diffusion_loss_binning.values(),
                                    diffusion_loss_binning_count.values(),
                                )
                            },
                        }
                    )

                diffusion_per_timestep = "\n\t".join(
                    [
                        f"{k}: {v / max(c, 1):.4f}"
                        for k, v, c in zip(
                            diffusion_loss_binning.keys(),
                            diffusion_loss_binning.values(),
                            diffusion_loss_binning_count.values(),
                        )
                    ]
                )

                logger.info(
                    "[Train] "
                    f"Epoch [{epoch}/{num_epochs}] "
                    f"Step [{global_step}/{max_steps}] "
                    f"Loss: {total_loss:.4f} "
                    f"Datapoints: {global_step * batch_size} "
                    # f"(Diff: {diffusion_loss:.4f}, "
                    f"LR: {lr_scheduler.get_last_lr()[0]:.6f}"
                    # f"\nDiffusion Per-timestep-binned:\n{diffusion_per_timestep}"
                )
                diffusion_loss_binning = {k: 0 for k in range(10)}
                diffusion_loss_binning_count = {k: 0 for k in range(10)}

            global_step += 1

            # Saving checkpoint
            if global_step % save_checkpoint_every == 0:
                save_weights(
                    logger, path_checkpoints, run_name, global_step, diffusion_model
                )
                # Evaluate samples
                evaluate_samples(
                    wandb,
                    global_step,
                    diffusion_model,
                    vae_model,
                    text_encoder,
                    tokenizer,
                    device,
                    seed=12
                )

            # Evaluation
            if global_step % evaluate_every == 1:
                generator = torch.Generator(device=device).manual_seed(12)

                val_diffusion_loss_binning = {k: 0 for k in range(10)}
                val_diffusion_loss_binning_count = {k: 0 for k in range(10)}
                diffusion_model.model.eval()

                total_losses = []
                for test_batch_idx, test_batch in enumerate(test_loader):
                    test_batch = (
                        test_batch[0].to(device, non_blocking=True), 
                        test_batch[1].to(device, non_blocking=True)
                    )
                    with torch.no_grad():
                        total_loss = forward(
                            diffusion_model=diffusion_model,
                            batch=test_batch,
                            vae_model=vae_model,
                            device=device,
                            global_step=global_step,
                            ctx=ctx,
                            generator=generator,
                            binnings=(
                                val_diffusion_loss_binning,
                                val_diffusion_loss_binning_count,
                            ),
                        )
                        total_losses.append(total_loss.item())

                    if test_batch_idx == 20:
                        break


                diffusion_model.model.train()
                total_loss = np.mean(total_losses).item()

                logger.info(
                    "[Eval] "
                    f"Epoch [{epoch}/{num_epochs}] "
                    f"Step [{global_step}/{max_steps}] "
                    f"Datapoints: {global_step * batch_size} "
                    f"Loss: {total_loss:.4f} "
                )

                if log_wandb:       
                    wandb.log(
                        {
                            "test/step": global_step,
                            "test/Datapoints": global_step * batch_size,
                            "test/total_loss": total_loss,
                            "test_binning/diffusion_loss_binning": {
                                k: v / max(c, 1)
                                for k, v, c in zip(
                                    val_diffusion_loss_binning.keys(),
                                    val_diffusion_loss_binning.values(),
                                    val_diffusion_loss_binning_count.values(),
                                )
                            },
                        }
                    )
                
    # Save last weights
    save_weights(
        logger, path_checkpoints, run_name, global_step, diffusion_model
    )
    
    # Cleanup
    if log_wandb:
        wandb.finish()


if __name__ == "__main__":
    train()