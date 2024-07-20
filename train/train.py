
from dotenv import load_dotenv
import sys
import os
import logging
from streaming import MDSWriter, LocalDataset
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

#####################################################33

PATH = '/mnt/sd1tb/tinydiffusion/dataset_v0/'
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

#####################################################

def base64_to_numpy(base64_str):
    decoded = base64.b64decode(base64_str)
    buffer = io.BytesIO(decoded)
    array = np.load(buffer)
    return array


class MosaicDataset(LocalDataset):
    def __init__(self, local, transform=None):
        super().__init__(local=local)
        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.Lambda(lambda img: img.convert('RGB')),
            torchvision.transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
            torchvision.transforms.RandomCrop(256),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.5], [0.5]),
        ])


    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        data['img'] = self.transforms(data['img'])
        data['t5_pool'] = base64_to_numpy(data['t5_pool'])
        data['vae_output'] = base64_to_numpy(data['vae_output'])
        return {
            'img': data['img'], 
            'vae': data['vae_output'], 
            't5_pool': data['t5_pool'],
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
    train_dataset = MosaicDataset(local=PATH + 'dataset/train/')
    val_dataset = MosaicDataset(local=PATH + 'dataset/val/')
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=64, pin_memory=True, num_workers=8, drop_last=True, shuffle=False
    )
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=64, pin_memory=True, num_workers=8, drop_last=True
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
        # run_name = '-'.join([
        #     date_str,
        #     f"model_{args.model_name}",
        #     f"m_{args.contrastive_method}",
        #     f"lr_{args.max_learning_rate}",
        #     f"b_{args.batch_size}",
        #     f"w_{args.num_workers}"
        # ])
        # args.run_name = run_name
        log_base_path = os.path.join(LOG_PATH, run_name)
        # Wandb
        if WANDB_LOG:
            logging.debug('Starting wandb.')
            wandb.init(
                project=os.getenv('WANDB_PROJECT'), 
                entity= os.getenv('WANDB_ENTITY'), 
                dir=LOG_PATH,
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
        monitor="val_loss", mode="min", every_n_epochs=1, save_on_train_epoch_end=True, dirpath=LOG_PATH
    )
    callbacks = [lr_monitor_callback, checkpoint_callback, log_image_callback]
    trainer = L.Trainer(
        num_sanity_val_steps=0,
        limit_train_batches=60,
        devices=1, 
        logger=logger,
        log_every_n_steps=25,
        callbacks=callbacks, #type: ignore
        precision='16-mixed',
        accumulate_grad_batches=1,
        gradient_clip_val=1,
        max_epochs=10,
        # enable_checkpointing=False
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


if __name__ == '__main__':
    run_training(sys.argv[1:])