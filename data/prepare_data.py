import torch
import webdataset as wds
from PIL import Image
import os
import tarfile
import io
import json
import click
import numpy as np
import matplotlib.pyplot as plt
import polars as pl
from torchvision import transforms
from dotenv import load_dotenv, find_dotenv
from diffusers import AutoencoderKL
from transformers import T5EncoderModel, T5TokenizerFast, AutoTokenizer, AutoModelForSeq2SeqLM
from huggingface_hub import login
from tqdm import tqdm

torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.unique_kernel_names = True
torch._inductor.config.fx_graph_cache = True

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

load_dotenv(find_dotenv('.env'))
login(os.getenv('HF_TOKEN'))

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WDS_VERBOSE_CACHE"] = "1"
os.environ["GOPEN_VERBOSE"] = "0"


def load_encoders(
    vae_path="black-forest-labs/FLUX.1-dev",
    text_encoder_path="black-forest-labs/FLUX.1-dev",
    device="cuda",
    compile_models=True,
):

    vae_model = (
        AutoencoderKL.from_pretrained(
            vae_path, torch_dtype=torch.float32, subfolder="vae"
        )
        .to(device)
        .eval()
    )
    vae_model.to(memory_format=torch.channels_last)
    vae_model.requires_grad_(False)

    # tokenizer = T5TokenizerFast.from_pretrained(
    #     text_encoder_path, subfolder="tokenizer_2"
    # )
    # text_encoder = (
    #     T5EncoderModel.from_pretrained(
    #         text_encoder_path,
    #         subfolder="text_encoder_2",
    #         torch_dtype=torch.float16,
    #     )
    #     .to(device)
    #     .eval()
    # )
    tokenizer = AutoTokenizer.from_pretrained(
        "EleutherAI/pile-t5-base", use_fast=True
    )
    tokenizer.pad_token = tokenizer.unk_token
    text_encoder = (
        AutoModelForSeq2SeqLM.from_pretrained(
            text_encoder_path,
            torch_dtype=torch.bfloat16,
        )
        .to(device)
        .eval()
    ).encoder
    text_encoder.requires_grad_(False)

    if compile_models:
        # vae_model.encode = torch.compile(vae_model.encode, mode="reduce-overhead")
        text_encoder.forward = torch.compile(
            text_encoder.forward, dynamic=True
        )

    return vae_model, tokenizer, text_encoder


def encode_prompt_with_t5(
    text_encoder,
    tokenizer,
    max_sequence_length=512,
    prompt=None,
    num_images_per_prompt=1,
    device=None,
    text_input_ids=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
    else:
        if text_input_ids is None:
            raise ValueError(
                "text_input_ids must be provided when the tokenizer is not specified"
            )
    
    prompt_embeds = text_encoder(text_input_ids.to(device))[0]

    dtype = text_encoder.dtype
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
    _, seq_len, _ = prompt_embeds.shape

    # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds


def build_dataset(base_path):
    transform_for_vae = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            transforms.Resize((256)),
            transforms.CenterCrop((256, 256)),
        ]
    )
    def transform_data(data):
        image, json_data = data
        image_vae = transform_for_vae(image)
        return image_vae, json_data["caption"]

    dataset = (
        wds.WebDataset(
            f"{base_path}/{{00000..00359}}.tar", 
            shardshuffle=False,
            nodesplitter=wds.split_by_node, 
            workersplitter=wds.split_by_worker
        )
        .decode("rgb", handler=wds.warn_and_continue)
        .to_tuple("jpg;png", "json", handler=wds.warn_and_continue)
        .map(transform_data)
    )
    return dataset
    

@click.command()
@click.option('--path_image_dataset', type=str, help='Path to the image dataset')
@click.option('--path_output_dataset', type=str, help='Path to save the dataset')
@click.option('--batch_size', type=int, help='Batch size', default=64)
def save_dataset(
    path_image_dataset,
    path_output_dataset,
    batch_size,
):
    dataset = build_dataset(path_image_dataset)

    torch.cuda.empty_cache()
    device = 'cuda'
    loader = wds.WebLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=16,
        pin_memory=True,
    )
    vae_model, tokenizer, text_encoder = load_encoders(
        vae_path="black-forest-labs/FLUX.1-dev",
        text_encoder_path="EleutherAI/pile-t5-large",
        device=device,
        compile_models=True,
    )

    os.makedirs(path_output_dataset, exist_ok=True)
    num_batches_per_file = 500
    
    sink = wds.ShardWriter(f"{path_output_dataset}/%06d.tar", maxcount=batch_size * num_batches_per_file)
    for idx, batch in tqdm(enumerate(loader)):            
        images_vae, captions = batch
        captions = [c for c in captions]
        
        # Generate data
        with torch.inference_mode():
            images_vae = images_vae.to(device, dtype=torch.float32)
            vae_latent = vae_model.encode(images_vae).latent_dist.sample()
            vae_latent = vae_latent.to(dtype=torch.bfloat16)
            
        with torch.inference_mode(), torch.amp.autocast(device_type=device, dtype=torch.bfloat16):
            prompt_embeds = encode_prompt_with_t5(
                text_encoder, tokenizer, prompt=captions, device=device, max_sequence_length=128
            )

        for i in range(len(captions)):
            key = f"{idx:06d}_{i}"
            sample = {
                '__key__': key,
                # 'img.pth': images_vae[i].cpu(),
                'vae_latent.pth': vae_latent[i].cpu(),
                'prompt_embeds.pth': prompt_embeds[i].cpu(),
                'caption.txt': captions[i]
            }
            sink.write(sample)

    sink.close()


if __name__ == "__main__":
    save_dataset()

# pyth