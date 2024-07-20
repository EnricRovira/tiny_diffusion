
import torch
import torchvision
import math
import wandb
import numpy as np
import lightning.pytorch as L
from lightning import Callback
from diffusers import AutoencoderKL, DDPMScheduler #type: ignore
from model.model import DiT_Llama



class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, base_lr, warmup_length, total_steps, last_epoch=-1, verbose=False):
        self.base_lr = base_lr
        self.warmup_length = warmup_length
        self.total_steps = total_steps
        super(CosineWarmupScheduler, self).__init__(optimizer, last_epoch, verbose)

    def _warmup_lr(self, step, base_lr):
        return base_lr * (step / self.warmup_length)

    def get_lr(self):
        step = self.last_epoch

        if step < self.warmup_length:
            lrs = [self._warmup_lr(step, group['initial_lr']) for group in self.optimizer.param_groups]
        else:
            e = step - self.warmup_length
            es = self.total_steps - self.warmup_length
            lrs = [0.5 * (1 + math.cos(math.pi * e / es)) * group['initial_lr'] for i, group in enumerate(self.optimizer.param_groups)]

        return lrs


class Trainer(L.LightningModule):
    '''
    Trainer of the GPT model, it shares a step function for both training and validation steps.
    config params saved into object
    '''
    def __init__(self, model_params=None, **kwargs):
        super().__init__()
        self.save_hyperparameters(model_params)
        self.noise_scheduler = DDPMScheduler.from_pretrained("dataautogpt3/PixArt-Sigma-900M", subfolder="scheduler")
        self.model = DiT_Llama(
            in_channels=16, input_size=32, patch_size=2, 
            dim=512, n_layers=16, n_heads=32
        )
        self.vae = AutoencoderKL.from_pretrained("ostris/vae-kl-f8-d16", torch_dtype=torch.float16) 
        self.vae.requires_grad_(False) #type: ignore
        self.vae_scaling_factor = self.vae.config.scaling_factor #type: ignore
        # Loss fn
        self.loss_fn = torch.nn.MSELoss()

    def configure_optimizers(self):  
        lr, wd = 2e-4, 0.01
        optimizer = torch.optim.AdamW(
            params=self.parameters(),
            lr=lr, 
            betas=(0.9, 0.95),
            weight_decay=wd,
            eps=1e-8,
        )

        lr_scheduler = CosineWarmupScheduler(
            optimizer, 
            lr, 
            warmup_length=int(0.1*(self.trainer.estimated_stepping_batches)), 
            total_steps=self.trainer.estimated_stepping_batches,
        )

        lr_scheduler_config = {
            "scheduler": lr_scheduler,
            "interval": "step",
            "frequency": 1,
            "name" : "lr_monitor"
        }
        return {'optimizer' : optimizer, 'lr_scheduler' : lr_scheduler_config}
    
    
    def forward(self, latent, timestep, cap_feats):
        output = self.model(latent, timestep, cap_feats)
        return output
    

    def training_step(self, batch):
        vae_latents, cap_feats = batch['vae'], batch['t5_pool']
        vae_latents = vae_latents * self.vae_scaling_factor
        bsz = vae_latents.shape[0]

        noise = torch.randn_like(vae_latents)
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=vae_latents.device)
        timesteps = timesteps.long()
        noisy_latents = self.noise_scheduler.add_noise(vae_latents, noise, timesteps)
        
        ######
        pred_latent = self(noisy_latents, timesteps, cap_feats)
        loss = self.loss_fn(pred_latent, noise).mean()
        ######
        self.log("train_loss", loss, prog_bar=True)
        return {
            "loss": loss,
        }


    def validation_step(self, batch, batch_idx):
        vae_latents, cap_feats = batch['vae'], batch['t5_pool']
        vae_latents = vae_latents * self.vae_scaling_factor
        bsz = vae_latents.shape[0]

        noise = torch.randn_like(vae_latents)
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=vae_latents.device)
        timesteps = timesteps.long()
        noisy_latents = self.noise_scheduler.add_noise(vae_latents, noise, timesteps)
        
        ######
        pred_latent = self(noisy_latents, timesteps, cap_feats)
        loss = self.loss_fn(pred_latent, noise).mean()
        ######
        self.log("val_loss", loss, prog_bar=True)
        # if batch_idx == 0:
        #     return {
        #         "loss": loss,
        #         "noisy_latents": vae_latents,
        #         "timestep": timesteps,
        #         "predictions": pred_latent
        #     }
        # else:
        return {
            "loss": loss,
        }

    def shared_epoch_end(self, outputs, stage):
        # aggregate step metics
        loss = torch.cat([torch.tensor([x["loss"]]) for x in outputs]).mean()
        # noisy_latents = torch.cat([torch.tensor([x["noisy_latents"]]) for x in outputs])
        # timesteps = torch.cat([torch.tensor([x["timesteps"]]) for x in outputs])
        # predictions = torch.cat([torch.tensor([x["predictions"]]) for x in outputs])
        # if stage != 'train':
            # self.log_validation_images(noisy_latents, timesteps, predictions)
        self.log_dict({
            f'{stage}_loss' : loss,
        }, prog_bar=True, on_epoch=True)


    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")  

    def test_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "test")
    



class LogPredictionsCallback(Callback):
    def __init__(self, val_dataloader):
        super().__init__()
        self.val_dataloader = val_dataloader

    def on_validation_epoch_end(self, trainer, pl_module):
        batch = next(iter(self.val_dataloader))
        caption = batch['caption']
        vae_latents, cap_feats = batch['vae'], batch['t5_pool']
        bsz = vae_latents.shape[0]
        device = pl_module.device

        # Build inputs
        vae_latents = vae_latents.to(device)
        cap_feats = cap_feats.to(device)
        noise = torch.randn_like(vae_latents, device=device)
        timesteps = torch.randint(0, pl_module.noise_scheduler.config.num_train_timesteps, (bsz,), device=device)
        timesteps = timesteps.long()
        noisy_latents = pl_module.noise_scheduler.add_noise(vae_latents, noise, timesteps)

        with torch.cuda.amp.autocast(), torch.no_grad():
            predictions = pl_module(noisy_latents, timesteps, cap_feats)
            noisy_latents_decoded = pl_module.vae.decode(noisy_latents).sample * pl_module.vae_scaling_factor #type: ignore
            predictions_decoded = pl_module.vae.decode(predictions).sample / pl_module.vae_scaling_factor #type: ignore

        num_images = min(16, bsz)
        images_to_log = []
        noisy_latents_decoded = np.transpose(noisy_latents_decoded.cpu().numpy(), (0, 2, 3, 1))
        predictions_decoded = np.transpose(predictions_decoded.cpu().numpy(), (0, 2, 3, 1)) 
        
        for i in range(num_images):
            _cap = caption[i]
            _noisy_latents_decoded = noisy_latents_decoded[i] 
            _predictions_decoded = predictions_decoded[i]
            images_to_log.append(wandb.Image(_noisy_latents_decoded, caption=f"IN-{_cap}"))
            images_to_log.append(wandb.Image(_predictions_decoded, caption=f"OUT-{_cap}"))

        trainer.logger.experiment.log({ #type: ignore
            "predictions": images_to_log,
            "epoch": trainer.current_epoch
        }, commit=False) 