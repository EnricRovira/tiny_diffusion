
import math
import torch 
from torch.nn.parallel import DistributedDataParallel as DDP 

class RectifiedFlow(torch.nn.Module):
    def __init__(self, model, ln=True, learn_sigma=False):
        super().__init__()
        self.model = model
        self.ln = ln
        self.stratified = False
        self.learn_sigma = learn_sigma

    def __call__(self, x, cond):
        b = x.size(0)
        if self.ln:
            if self.stratified:
                # stratified sampling of normals
                # first stratified sample from uniform
                quantiles = torch.linspace(0, 1, b + 1).to(x.device)
                z = quantiles[:-1] + torch.rand((b,)).to(x.device) / b
                # now transform to normal
                z = torch.erfinv(2 * z - 1) * math.sqrt(2)
                t = torch.sigmoid(z)
            else:
                nt = torch.randn((b,)).to(x.device)
                t = torch.sigmoid(nt)
        else:
            t = torch.rand((b,)).to(x.device)
        texp = t.view([b, *([1] * len(x.shape[1:]))])
        z1 = torch.randn_like(x)
        zt = (1 - texp) * x + texp * z1

        # make t, zt into same dtype as x
        zt, t = zt.to(x.dtype), t.to(x.dtype)
        vtheta = self.model(zt, cond, t) 
        if self.learn_sigma: 
            vtheta, _ = vtheta.chunk(2, dim=1) 
        batchwise_mse = (
            # (z1 - x - vtheta) ** 2).mean(dim=list(range(1, len(x.shape)))
            (z1.float() - x.float() - vtheta.float()).pow(2).mean(dim=(1, 2, 3))
        )
        # tlist = batchwise_mse.detach().cpu().reshape(-1).tolist()
        # ttloss = [(tv, tloss) for tv, tloss in zip(t, tlist)]
        return batchwise_mse.mean(), {'output': vtheta}

    @torch.no_grad()
    def sample(self, z, cond, null_cond=None, sample_steps=50, cfg=2.0):
        b = z.size(0)
        dt = 1.0 / sample_steps
        dt = torch.tensor([dt] * b).to(z.device).view([b, *([1] * len(z.shape[1:]))])
        images = [z]
        for i in range(sample_steps, 0, -1):
            t = i / sample_steps
            t = torch.tensor([t] * b).to(z.device)

            vc = self.model(z, cond, t) 
            if self.learn_sigma: 
                vc, _ = vc.chunk(2, dim=1) 
            if null_cond is not None:
                vu = self.model(z, null_cond, t) 
                if self.learn_sigma: 
                    vu, _ = vu.chunk(2, dim=1) 
                vc = vu + cfg * (vc - vu)

            z = z - dt * vc
            images.append(z)
        return images


    @torch.no_grad()
    def sample_with_xps(self, z, cond, null_cond=None, sample_steps=50, cfg=2.0):
        b = z.size(0)
        dt = 1.0 / sample_steps
        dt = torch.tensor([dt] * b).to(z.device).view([b, *([1] * len(z.shape[1:]))])
        images = [z]
        for i in range(sample_steps, 0, -1):
            t = i / sample_steps
            t = torch.tensor([t] * b).to(z.device)

            vc = self.model(z, cond, t) 
            if self.learn_sigma: 
                vc, _ = vc.chunk(2, dim=1)  
            if null_cond is not None:
                vu = self.model(z, null_cond, t) 
                if self.learn_sigma: 
                    vu, _ = vu.chunk(2, dim=1) 
                vc = vu + cfg * (vc - vu)
            x = z - i * dt * vc
            z = z - dt * vc
            images.append(x)
        return images