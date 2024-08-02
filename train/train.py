
from dotenv import load_dotenv
import sys
import os
import random
import logging
from streaming import MDSWriter, LocalDataset, StreamingDataset
import base64 
import io
import wandb
import numpy as np
import torch
import torchvision
import lightning.pytorch as L
from lightning.pytorch.loggers import WandbLogger, CSVLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from torch.utils.data import Dataset, DataLoader
from dataclasses import asdict
from datetime import datetime
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
load_dotenv()

from model.model import DiT_Llama
from model.trainer import Trainer, LogPredictionsCallback
import warnings

os.environ['TOKENIZERS_PARALLELISM'] = "False"

#####################################################33

PATH = '/mnt/sd1tb/tinydiffusion/dataset_v1/'
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
DEVICE = "cuda" if torch.cuda.is_available() else None
if not DEVICE:
    logging.error('CUDA DEVICE NOT AVAILABLE') 

SEED = 12
WANDB_LOG = True
DEBUG = False
LOG_PATH = './logs/'
os.makedirs(LOG_PATH, exist_ok=True)
warnings.filterwarnings("ignore", category=UserWarning)

#####################################################

def base64_to_numpy(base64_str):
    decoded = base64.b64decode(base64_str)
    buffer = io.BytesIO(decoded)
    array = np.load(buffer)
    return array


def get_uncond_emb():
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pile-t5-base", use_fast=True)
    tokenizer.pad_token = tokenizer.bos_token
    t5model = AutoModelForSeq2SeqLM.from_pretrained("EleutherAI/pile-t5-base", torch_dtype=torch.float16).to(DEVICE)
    t5model.requires_grad_(False)
    uncond_caption = tokenizer(
        [""],   
        truncation=True,
        max_length=128, 
        padding="max_length", 
        return_tensors="pt"
    )
    uncond_caption['attention_mask'] = uncond_caption["attention_mask"].unsqueeze(0) #type: ignore
    inputs = {k: v.to(DEVICE) for k, v in uncond_caption.items()}
    with torch.amp.autocast(DEVICE), torch.inference_mode(): #type: ignore
        out = t5model.encoder(**inputs)[0]
    out_mask = inputs["attention_mask"].permute(0, 2, 1).expand(out.shape)
    out = (out * out_mask).float().cpu().numpy()

    del t5model
    torch.cuda.empty_cache()

    return out[0, :, :]


class MosaicDataset(StreamingDataset):
    def __init__(self, local, batch_size=1, shuffle=False, prob_uncond: float=0.1):
        super().__init__(
            local=local, batch_size=batch_size, 
            predownload=16*batch_size, 
            shuffle=shuffle,
            num_canonical_nodes=128 * 32,
            shuffle_block_size=1 << 18,
            keep_zip=False
        )
        self.prob_uncond = prob_uncond
        self.uncond_emb = get_uncond_emb()
        

    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        # data['img'] = self.transforms(data['img'])
        # data['t5_pool'] = base64_to_numpy(data['t5_pool'])
        # data['t5_output'] = data['t5_output']
        # data['vae_output'] = data['vae_output']
        if random.random() <= self.prob_uncond:
            text_encoder = self.uncond_emb

        else:
            text_encoder = data['t5_output'][:128, :]
        return {
            'vae': data['vae_output'], 
            'text_encoder': text_encoder,
            'caption': data['caption']
        }
    

def run_training(args):
    print(args)
    logging.info('Starting training...')
    L.seed_everything(SEED)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Get data
    bs = 192
    train_dataset = MosaicDataset(local=PATH + 'dataset/train/', batch_size=bs, shuffle=False)
    val_dataset = MosaicDataset(local=PATH + 'dataset/val/', batch_size=bs, shuffle=False)
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=bs, pin_memory=True, num_workers=14, drop_last=True,
    )
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=bs, pin_memory=True, num_workers=14, drop_last=True
    )

    # Get Model
    model = Trainer()
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Number of parameters: {num_params / 1e6}M")

    # Logging
    if not args or args.get('run_name') is None:
        logging.debug('Getting log name.')
        date_str = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        run_name = date_str
        log_base_path = os.path.join(LOG_PATH, run_name)
        os.makedirs(log_base_path, exist_ok=True)
        # Wandb
        if WANDB_LOG:
            logging.debug('Starting wandb.')
            wandb.init(
                project=os.getenv('WANDB_PROJECT'), 
                entity= os.getenv('WANDB_ENTITY'), 
                dir=log_base_path,
                name=run_name,
            )
            logger = WandbLogger(log_model="all")
            if DEBUG:
                wandb.watch(model, log='all')
            wandb_id = wandb.run.id #type: ignore
            logging.debug('Finished loading wandb.')
        else:
            logger = CSVLogger(save_dir=log_base_path)


    # Trainer object
    lr_monitor_callback = LearningRateMonitor(logging_interval='step')
    log_image_callback = LogPredictionsCallback(val_dataloader=val_dataloader)
    checkpoint_callback = ModelCheckpoint(
        every_n_epochs=2, 
        save_on_train_epoch_end=True, 
        dirpath=log_base_path
    )
    callbacks = [lr_monitor_callback, checkpoint_callback, log_image_callback]
    trainer = L.Trainer(
        num_sanity_val_steps=0,
        # limit_train_batches=20,
        devices=1, 
        logger=logger,
        log_every_n_steps=50,
        callbacks=callbacks, #type: ignore
        precision='16-mixed',
        accumulate_grad_batches=1,
        gradient_clip_val=0.5,
        max_epochs=20,
    )
    # logging.info(f"Compiling model...")
    # model = torch.compile(model)
    logging.info("Starting training...")
    trainer.fit(
        model=model, 
        train_dataloaders=train_dataloader, 
        val_dataloaders=val_dataloader
    )
    logging.info("Finished training.")
    wandb.finish()


if __name__ == '__main__':
    run_training(sys.argv[1:])