
import io
import os
import json
from torchvision import transforms
import webdataset as wds
from diffusers import AutoencoderKL
from transformers import T5EncoderModel, T5TokenizerFast, AutoTokenizer, AutoModelForSeq2SeqLM
from huggingface_hub import login
import torch


# def build_dataloader(
#     base_path, 
#     shuffle=False,
#     batch_size=32,
#     num_workers=2,
#     persistent_workers=False,
#     pin_memory=False,
#     cache_dir=None
# ):
#     def decode_sample(to_array=True):    
#         def _decode_sample(sample):  
#             # img_buffer = io.BytesIO(sample["image.pth"])
#             prompt_embeds_buffer = io.BytesIO(sample["prompt_embeds.pth"])
#             vae_latent_buffer = io.BytesIO(sample["vae_latent.pth"])
#             return dict(
#                 # img=torch.load(img_buffer, weights_only=True),
#                 prompt_embeds=torch.load(prompt_embeds_buffer, weights_only=True),
#                 vae_latent=torch.load(vae_latent_buffer, weights_only=True),
#                 metadata=json.loads(sample["json"])
#             )
#         return _decode_sample
    
#     if cache_dir:
#         os.makedirs(cache_dir, exist_ok=True)

#     dataset = (
#         wds.WebDataset(
#             base_path, 
#             shardshuffle=False,
#             # resampled=True,
#             nodesplitter=wds.split_by_node, 
#             workersplitter=wds.split_by_worker,
#             cache_dir=cache_dir,
#         )
#         .map(decode_sample(to_array=False), handler=wds.warn_and_continue)
#     )
#     if shuffle:
#         dataset = dataset.shuffle(1_000)

#     # dataset = dataset.batched(batch_size)
#     # loader = wds.WebLoader(dataset, batch_size=None, num_workers=num_workers)
#     loader = wds.WebLoader(
#         dataset,
#         batch_size=batch_size,
#         shuffle=False,
#         num_workers=num_workers,
#         pin_memory=pin_memory,
#         persistent_workers=persistent_workers,
#     )
#     # loader = loader.unbatched().shuffle(1_000).batched(batch_size)
#     return loader

def build_dataloader(
    base_path, 
    shuffle=False,
    batch_size=32,
    num_workers=2,
    persistent_workers=False,
    pin_memory=False,
    resampled=True,
    max_items_per_epoch=100_000
):
    def decode_sample(key, data):
        if key.endswith('.pth'):
            return torch.load(io.BytesIO(data), weights_only=False)
        elif key.endswith('.txt'):
            return data.decode('utf-8')
        else:
            return data
        
    def make_sample(sample):
        latent = sample['vae_latent.pth']
        prompt_embeds = sample['prompt_embeds.pth']
        return latent, prompt_embeds

    dataset = (
        wds.WebDataset(
            base_path, 
            shardshuffle=False,
            nodesplitter=wds.split_by_node, 
            workersplitter=wds.split_by_worker,
            resampled=resampled # Makes a whole iteration over the dataset files and repeats it again
        )
        .decode(decode_sample)
        .map(make_sample)
    )
    # Shuffle between files
    if shuffle:
        dataset = dataset.shuffle(1_000)

    dataset = dataset.batched(batch_size)
    dataloader = wds.WebLoader(
        dataset, batch_size=None, num_workers=num_workers,
    )
    # Shuffle between workers
    dataloader = dataloader.unbatched()
    if shuffle:
        dataloader = dataloader.shuffle(1_024)
    dataloader = dataloader.batched(batch_size)
    dataloader = dataloader.with_epoch(max_items_per_epoch // batch_size)
    #     dataloader = torch.utils.data.DataLoader(
    #     dataset,
    #     batch_size=batch_size,
    #     num_workers=num_workers,
    #     pin_memory=pin_memory,
    #     persistent_workers=persistent_workers,
    #     # collate_fn=wds.collation.default_collate  # Use default collate function
    # )
    return dataloader

def load_encoders(
    vae_path="black-forest-labs/FLUX.1-dev",
    text_encoder_path="EleutherAI/pile-t5-base",
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
        text_encoder_path, use_fast=True
    )
    tokenizer.pad_token = tokenizer.unk_token
    text_encoder = (
        AutoModelForSeq2SeqLM.from_pretrained(
            text_encoder_path,
            torch_dtype=torch.float16,
        )
        .to(device)
        .eval()
    ).encoder
    text_encoder.requires_grad_(False)

    if compile_models:
        # vae_model.encode = torch.compile(vae_model.encode, mode="reduce-overhead")
        text_encoder = torch.compile(
            text_encoder#, mode="reduce-overhead"
        )

    return vae_model, tokenizer, text_encoder