
import torch
import torchvision
import math
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
import wandb
import numpy as np
import lightning.pytorch as L
from lightning import Callback
from model.model import DiT_Llama, DiT_Llama_B, DiT_Llama_S
from diffusers import AutoencoderKL, DDPMScheduler #type: ignore
from diffusers.image_processor import VaeImageProcessor



class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, base_lr, warmup_length, total_steps, last_epoch=-1, verbose=False):
        self.base_lr = base_lr
        self.warmup_length = warmup_length
        self.total_steps = total_steps
        super(CosineWarmupScheduler, self).__init__(
            optimizer, 
            last_epoch, 
            verbose #type: ignore
        )

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
        # self.model = DiT_Llama(
            # in_channels=16, input_size=32, patch_size=2, 
            # dim=768, n_layers=12, n_heads=12
        # )
        self.model = DiT_Llama_S()
        self.vae = AutoencoderKL.from_pretrained("ostris/vae-kl-f8-d16", torch_dtype=torch.float16) 
        self.vae.requires_grad_(False) #type: ignore
        self.vae_scaling_factor = self.vae.config.scaling_factor #type: ignore
        self.loss_fn = torch.nn.MSELoss()

    def configure_optimizers(self):  
        lr, wd = 7e-5, 0.01
        optimizer = torch.optim.AdamW( #type: ignore
            params=self.parameters(),
            lr=lr, 
            betas=(0.9, 0.95),
            weight_decay=wd,
            eps=1e-8,
        )

        lr_scheduler = CosineWarmupScheduler(
            optimizer, 
            lr, 
            warmup_length=2500,#int(0.01*(self.trainer.estimated_stepping_batches)), 
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
        vae_latents, cap_feats = batch['vae'], batch['text_encoder']
        vae_latents = vae_latents * self.vae_scaling_factor
        bsz = vae_latents.shape[0]

        noise = torch.randn_like(vae_latents)
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=vae_latents.device
        ).long()
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
        vae_latents, cap_feats = batch['vae'], batch['text_encoder']
        vae_latents = vae_latents * self.vae_scaling_factor
        bsz = vae_latents.shape[0]

        noise = torch.randn_like(vae_latents)
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=vae_latents.device
        ).long()
        noisy_latents = self.noise_scheduler.add_noise(vae_latents, noise, timesteps)
        
        ######
        pred_latent = self(noisy_latents, timesteps, cap_feats)
        loss = self.loss_fn(pred_latent, noise).mean()
        ######
        self.log("val_loss", loss, prog_bar=True)
        return {
            "loss": loss,
        }

    def shared_epoch_end(self, outputs, stage):
        # aggregate step metics
        loss = torch.cat([torch.tensor([x["loss"]]) for x in outputs]).mean()
        self.log_dict({
            f'{stage}_loss' : loss,
        }, prog_bar=True, on_epoch=True)


    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")  

    def test_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "test")
    



class LogPredictionsCallback(Callback):
    def __init__(self, val_dataloader, num_imgs=8, num_inference_steps=30):
        super().__init__()
        self.val_dataloader = val_dataloader
        self.num_imgs = num_imgs
        self.num_inference_steps = num_inference_steps


    def generate_img_grid(self, imgs):
        num_imgs = len(imgs)
        filas = math.ceil(math.sqrt(num_imgs))
        columnas = math.ceil(num_imgs / filas)
        
        fig, axes = plt.subplots(filas, columnas, figsize=(columnas * 3, filas * 3))
        axes = axes.flatten() #type: ignore

        for i in range(len(axes)):
            if i < num_imgs:
                axes[i].imshow(imgs[i])
                axes[i].set_title(f'Img {i + 1}', fontsize=12)
            else:
                axes[i].axis('off')
            axes[i].axis('off')

        plt.tight_layout()

        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig)
        

        img_pil = Image.open(buf)
        return img_pil 

    
    def on_validation_epoch_end(self, trainer, pl_module):
        torch.cuda.empty_cache()
        batch = next(iter(self.val_dataloader))
        device = pl_module.device

        # Build scheduler
        scheduler = pl_module.noise_scheduler
        scheduler.set_timesteps(num_inference_steps=self.num_inference_steps)
        
        # Build inputs
        images_to_log = []
        raw_bsz = batch['vae'].shape[0]
        num_images = min(self.num_imgs, raw_bsz)
        captions = batch['caption'][:num_images]
        cap_feats = batch['text_encoder'][:num_images].to(device)
        for idx_img in range(num_images):
            list_imgs = []
            x = torch.randn(1, 16, 32, 32).to(device)
            _cap_feats = cap_feats[idx_img].unsqueeze(0)
            for i, t in enumerate(scheduler.timesteps):
                x = x.to(device)
                t = t.unsqueeze(0).to(device)
                latent = scheduler.scale_model_input(x, timestep=t)
                with torch.amp.autocast("cuda"), torch.inference_mode(): #type: ignore
                    noise_pred = pl_module(latent, t, _cap_feats)
                x = scheduler.step(noise_pred, t, x).prev_sample.to(device)
                with torch.amp.autocast("cuda"), torch.inference_mode(): #type: ignore
                    img = pl_module.vae.decode(x / pl_module.vae_scaling_factor).sample
                img = VaeImageProcessor().postprocess(
                    image=img, do_denormalize=[True, True]
                )[0] # type: ignore
                list_imgs.append(img)

            generated_grid = self.generate_img_grid(list_imgs)
            images_to_log.append(wandb.Image(generated_grid, caption=captions[idx_img]))

        trainer.logger.experiment.log({ #type: ignore
            "predictions": images_to_log,
            "epoch": trainer.current_epoch
        }, commit=False)
        torch.cuda.empty_cache()

    # def on_validation_epoch_end(self, trainer, pl_module):
    #     batch = next(iter(self.val_dataloader))
    #     raw_bsz = batch['vae'].shape[0]
    #     num_images = min(16, raw_bsz)
    #     batch = {k: v[:num_images] for k, v in batch.items()}
    #     caption = batch['caption']
    #     vae_latents, cap_feats = batch['vae'], batch['t5_pool']
    #     device = pl_module.device
    #     bsz = batch['vae'].shape[0]

    #     # Build inputs
    #     vae_latents = vae_latents.to(device)
    #     cap_feats = cap_feats.to(device)
    #     noise = torch.randn_like(vae_latents, device=device)
    #     timesteps = torch.randint(0, pl_module.noise_scheduler.config.num_train_timesteps, (bsz,), device=device)
    #     timesteps = timesteps.long()
    #     noisy_latents = pl_module.noise_scheduler.add_noise(vae_latents, noise, timesteps)

    #     with torch.cuda.amp.autocast(), torch.no_grad():
    #         predictions = pl_module(noisy_latents, timesteps, cap_feats)
    #         noisy_latents_decoded = pl_module.vae.decode(noisy_latents / pl_module.vae_scaling_factor).sample  #type: ignore
    #         predictions_decoded = pl_module.vae.decode(predictions / pl_module.vae_scaling_factor).sample  #type: ignore
        
    #     images_to_log = []   
    #     for i in range(num_images):
    #         _cap = caption[i]
    #         _noisy_latents_decoded = VaeImageProcessor().postprocess(
    #             image=noisy_latents_decoded[i].unsqueeze(0), do_denormalize=[True, True]
    #         )[0] #type: ignore
    #         _predictions_decoded = VaeImageProcessor().postprocess(
    #             image=predictions_decoded[i].unsqueeze(0), do_denormalize=[True, True]
    #         )[0] #type: ignore
    #         images_to_log.append(wandb.Image(_noisy_latents_decoded, caption=f"IN-{_cap}"))
    #         images_to_log.append(wandb.Image(_predictions_decoded, caption=f"OUT-{_cap}"))

    #     trainer.logger.experiment.log({ #type: ignore
    #         "predictions": images_to_log,
    #         "epoch": trainer.current_epoch
    #     }, commit=False)
    #     torch.cuda.empty_cache()