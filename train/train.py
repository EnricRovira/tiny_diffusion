
from dotenv import load_dotenv
import sys
import os
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
load_dotenv()

from model.model import DiT_Llama
from model.trainer import Trainer, LogPredictionsCallback
import warnings

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


class MosaicDataset(StreamingDataset):
    def __init__(self, local, batch_size=1, shuffle=False):
        super().__init__(
            local=local, batch_size=batch_size, 
            predownload=16*batch_size, 
            shuffle=shuffle,
            num_canonical_nodes=128 * 32,
            shuffle_block_size=1 << 18,
            keep_zip=False
        )

    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        # data['img'] = self.transforms(data['img'])
        # data['t5_pool'] = base64_to_numpy(data['t5_pool'])
        # data['t5_output'] = data['t5_output']
        # data['vae_output'] = data['vae_output']
        return {
            # 'img': data['img'], 
            'vae': data['vae_output'], 
            'text_encoder': data['t5_output'],
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
    bs = 128
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
        every_n_epochs=1, 
        save_on_train_epoch_end=True, 
        dirpath=log_base_path
    )
    callbacks = [lr_monitor_callback, checkpoint_callback, log_image_callback]
    trainer = L.Trainer(
        num_sanity_val_steps=0,
        # limit_train_batches=20,
        devices=1, 
        logger=logger,
        log_every_n_steps=30,
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