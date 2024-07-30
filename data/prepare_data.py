"""
Prepare data
"""

import os
import logging
import re
import shutil
import io
import numpy as np
import base64
import torch
import polars as pl
from PIL import Image
from tqdm import tqdm
import torchvision
from torchvision.transforms.functional import InterpolationMode
from torch.utils.data import Dataset, DataLoader
from diffusers.image_processor import VaeImageProcessor
from diffusers import AutoencoderKL, DDPMScheduler #type: ignore
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from streaming import MDSWriter, LocalDataset

pl.Config.set_fmt_str_lengths(200)
torch.cuda.empty_cache()
np.random.seed(12)
#####################################################33

PATH = '/mnt/sd1tb/tinydiffusion/dataset_v1/'
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
DEVICE = "cuda" if torch.cuda.is_available() else None
if not DEVICE:
    logging.error('CUDA DEVICE NOT AVAILABLE') 

#####################################################

image_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Lambda(lambda img: img.convert('RGB')),
    torchvision.transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
    torchvision.transforms.RandomCrop(256),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.5], [0.5]),
])


class TinyDiffusionDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform
        self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pile-t5-base", use_fast=True)
        self.tokenizer.pad_token = self.tokenizer.bos_token

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe[idx, "path"]
        caption = self.dataframe[idx, "caption"]
        image = Image.open(img_path)

        # Caption
        caption = self.tokenizer(
            caption, 
            truncation=True,
            max_length=77*3, 
            padding="max_length", 
            return_tensors="pt"
        )
        caption['input_ids'] = caption["input_ids"].squeeze() #type: ignore

        # Image
        if self.transform:
            image = self.transform(image)

        return {"idxs": idx, "image": image, "caption": caption}


def numpy_to_base64(array):
    buffer = io.BytesIO()
    np.save(buffer, array)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def process_batch(df, batch, vae, t5model, writer):
    t5_inputs = {k: v.to(DEVICE) for k, v in batch['caption'].items()}
    vae_input = batch['image'].to(DEVICE)
    with torch.amp.autocast(DEVICE): #type: ignore
        vae_output = vae.encode(vae_input).latent_dist.sample()
        t5_outputs = t5model.encoder(**t5_inputs)[0]
    t5_mask = t5_inputs["attention_mask"].permute(0, 2, 1).expand(t5_outputs.shape)
    t5_outputs = t5_outputs * t5_mask
    t5_pool = (t5_outputs).sum(1) / t5_mask.sum(dim=1)

    # Format outputs
    t5_outputs = t5_outputs.cpu().numpy()
    t5_mask = t5_mask.cpu().numpy()
    t5_pool = t5_pool.cpu().numpy()
    vae_output = vae_output.cpu().numpy()
    idxs = batch['idxs'].numpy()
    df_chunk = df[idxs]

    # Save output
    samples = [
        {
            'id': row['id'],
            # 'img': Image.open(row['path']),
            'caption': row['caption'],
            # 'original_width': row['original_width'],
            # 'original_height': row['original_height'],
            # 'width': row['width'],
            # 'height': row['height'],
            'vae_output': vae_output[i],
            'vae_scaling_factor': vae.config.scaling_factor,
            't5_output': t5_outputs[i],#numpy_to_base64(t5_outputs[i]),
            # 't5_mask': numpy_to_base64(t5_mask[i]),
            # 't5_pool': numpy_to_base64(t5_pool[i]),
        }
        for i, row in enumerate(df_chunk.iter_rows(named=True))
    ]

    for sample in samples:
        writer.write(sample)


def prepare_data(
    df_train,
    df_val, 
    path_output_dataset
):
    columns = {
        'id': 'str',
        # 'img': 'jpeg',
        'caption': 'str',
        # 'original_width': 'int',
        # 'original_height': 'int',
        # 'width': 'int',
        # 'height': 'int',
        'vae_output': 'ndarray',
        'vae_scaling_factor': 'float32',
        't5_output': 'ndarray',
        # 't5_mask': 'str',
        # 't5_pool': 'str',
    }

    # Load model and data
    vae = AutoencoderKL.from_pretrained("ostris/vae-kl-f8-d16", torch_dtype=torch.float16).to(DEVICE) #type: ignore
    _ = vae.requires_grad_(False)
    t5model = AutoModelForSeq2SeqLM.from_pretrained("EleutherAI/pile-t5-base", torch_dtype=torch.float16).to(DEVICE)
    t5model.requires_grad_(False)
    train_dataset = TinyDiffusionDataset(df_train, transform=image_transforms)
    val_dataset = TinyDiffusionDataset(df_val, transform=image_transforms)
    train_dataloader = DataLoader(train_dataset, batch_size=128, pin_memory=True, num_workers=10)
    val_dataloader = DataLoader(val_dataset, batch_size=128, pin_memory=True, num_workers=10)

    # Generate outputs
    compression = 'zstd:7'
    logging.info('Processng train...')
    with MDSWriter(
        out=path_output_dataset + 'train/', columns=columns, max_workers=16,
        exist_ok=True, progress_bar=True, size_limit=1 << 29, compression=compression
    ) as out:
        # Train
        for batch in tqdm(train_dataloader):
            process_batch(df_train, batch, vae, t5model, out)
    logging.info('Processng validation...')
    with MDSWriter(
        out=path_output_dataset + 'val/', columns=columns, max_workers=16,
        exist_ok=True, progress_bar=True, size_limit=1 << 29, compression=compression
    ) as out:
        # Val
        for batch in tqdm(val_dataloader):
            process_batch(df_val, batch, vae, t5model, out)
            


def build_splits(df):
    df = df.with_columns(
        pl.when(pl.col("source") != "real")
        .then(pl.lit("train"))
        .otherwise(pl.lit(None))
        .alias("set")
    )

    df_real = df.filter(pl.col('source')=='real')
    idxs = np.arange(len(df_real))
    np.random.shuffle(idxs)
    tr_idxs, val_idxs = idxs[:-1024], idxs[-1024:]
    df_train = pl.concat([
        df_real[tr_idxs],
        df.filter(pl.col('set')=='train')
    ])
    df_val = df_real[val_idxs]
    assert len(df_train)+len(df_val)==len(df)
    return df_train, df_val

def main():
    path_output_dataset = PATH + 'dataset/'
    df = pl.read_parquet(PATH + 'dataset_gold.parquet')
    df_train, df_val = build_splits(df)
    logging.info(
        f'Starting generation - Records all: {len(df)} - train_records: {len(df_train)} - val_records: {len(df_val)}'
    )
    prepare_data(df_train, df_val, path_output_dataset)


if __name__ == "__main__":
    main()

