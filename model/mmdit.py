# DiT model taken from: https://github.com/cloneofsimo/repa-rf/blob/main/model.py
# removed repa loss

import math
from collections import defaultdict
from tqdm import tqdm
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.unique_kernel_names = True
torch._inductor.config.fx_graph_cache = True

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def modulate(x, shift, scale):
    return x * (1 + scale) + shift

def find_multiple(n: int, k: int) -> int:
    if n % k == 0:
        return n
    return n + k - (n % k)


class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations."""

    def __init__(
        self, 
        hidden_size, 
        frequency_embedding_size=256, 
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(
                frequency_embedding_size,
                hidden_size,
                bias=True,
            ),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        if torch.is_floating_point(t):
            embedding = embedding.to(dtype=t.dtype)
        return embedding

    def forward(self, t, **kwargs):
        dtype = t.dtype
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size).to(dtype)
        t_emb = self.mlp(t_freq)
        return t_emb


class VectorEmbedder(nn.Module):
    """Embeds a flat vector of dimension input_dim"""

    def __init__(self, input_dim: int, hidden_size: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    @torch.compile()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)
    

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()           
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    @torch.compile()
    def forward(self, x):
        x_dtype = x.dtype
        x = x.float()
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * norm * self.weight).to(dtype=x_dtype)


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim=None) -> None:
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 4 * dim

        n_hidden = int(2 * hidden_dim / 3)
        n_hidden = find_multiple(n_hidden, 256)

        self.c_fc1 = nn.Linear(dim, n_hidden, bias=False)
        self.c_fc2 = nn.Linear(dim, n_hidden, bias=False)
        self.c_proj = nn.Linear(n_hidden, dim, bias=False)

    @torch.compile()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.c_fc1(x)) * self.c_fc2(x)
        x = self.c_proj(x)
        return x


def split_qkv(qkv, head_dim):
    qkv = qkv.reshape(qkv.shape[0], qkv.shape[1], 3, -1, head_dim).movedim(2, 0)
    return qkv[0], qkv[1], qkv[2]

def attention(q, k, v, heads, mask=None):
    """Convenience wrapper around a basic attention operation"""
    b, _, dim_head = q.shape
    dim_head //= heads
    q, k, v = map(lambda t: t.view(b, -1, heads, dim_head).transpose(1, 2), (q, k, v))
    out = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False
    )
    return out.transpose(1, 2).reshape(b, -1, heads * dim_head)


class SelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.ln_q = RMSNorm(self.head_dim)
        self.ln_k = RMSNorm(self.head_dim)

    def pre_attention(self, x: torch.Tensor):
        B, L, C = x.shape
        qkv = self.qkv(x)
        q, k, v = split_qkv(qkv, self.head_dim)
        q = self.ln_q(q).reshape(q.shape[0], q.shape[1], -1)
        k = self.ln_k(k).reshape(q.shape[0], q.shape[1], -1)
        return (q, k, v)

    def post_attention(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return x

    @torch.compile()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        (q, k, v) = self.pre_attention(x)
        x = attention(q, k, v, self.num_heads)
        x = self.post_attention(x)
        return x
    

class DiTBlock(nn.Module):
    def __init__(self, dim, heads=8, global_conddim=1024):
        super().__init__()

        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)

        self.modCX = nn.Sequential(
            nn.SiLU(),
            nn.Linear(global_conddim, 6 * dim, bias=False),
        )

        self.attn = SelfAttention(dim, heads)
        self.mlp = MLP(dim, hidden_dim=dim * 4)

    @torch.compile()
    def forward(self, cx, global_cond, **kwargs):
        cxres = cx
        (
            shift_msa, scale_msa, gate_msa, 
            shift_mlp, scale_mlp, gate_mlp
        ) = self.modCX(global_cond).chunk(6, dim=1)

        cx = modulate(self.norm1(cx), shift_msa[:, None, :], scale_msa[:, None, :])
        cx = self.attn(cx)
        cx = self.norm2(cxres + gate_msa.unsqueeze(1) * cx)
        mlpout = self.mlp(modulate(cx, shift_mlp[:, None, :], scale_mlp[:, None, :]))
        cx = gate_mlp.unsqueeze(1) * mlpout

        cx = cxres + cx

        return cx

class CrossAttention(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()

        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        # cond
        self.w1qkv = nn.Linear(dim, dim * 3, bias=False)
        self.w1o = nn.Linear(dim, dim, bias=False)

        # x
        self.w2qkv = nn.Linear(dim, dim * 3, bias=False)
        self.w2o = nn.Linear(dim, dim, bias=False)

        self.q_norm1 = RMSNorm(self.head_dim)
        self.k_norm1 = RMSNorm(self.head_dim)

        self.q_norm2 = RMSNorm(self.head_dim)
        self.k_norm2 = RMSNorm(self.head_dim)


    @torch.compile()
    def forward(self, c, x):
        bsz, seqlen1, _ = c.shape
        bsz, seqlen2, _ = x.shape
        seqlen = seqlen1 + seqlen2


        cqkv = self.w1qkv(c)
        cq, ck, cv = split_qkv(cqkv, self.head_dim)
        # cq = cq.view(bsz, seqlen1, self.n_heads, self.head_dim)
        # ck = ck.view(bsz, seqlen1, self.n_heads, self.head_dim)
        # cv = cv.view(bsz, seqlen1, self.n_heads, self.head_dim)
        cq, ck = self.q_norm1(cq), self.k_norm1(ck)

        xqkv = self.w2qkv(x)
        xq, xk, xv = split_qkv(xqkv, self.head_dim)
        # xq = xq.view(bsz, seqlen2, self.n_heads, self.head_dim)
        # xk = xk.view(bsz, seqlen2, self.n_heads, self.head_dim)
        # xv = xv.view(bsz, seqlen2, self.n_heads, self.head_dim)
        xq, xk = self.q_norm2(xq), self.k_norm2(xk)

        # concat all
        q, k, v = (
            torch.cat([cq, xq], dim=1),
            torch.cat([ck, xk], dim=1),
            torch.cat([cv, xv], dim=1),
        )

        output = F.scaled_dot_product_attention(
            q.permute(0, 2, 1, 3),
            k.permute(0, 2, 1, 3),
            v.permute(0, 2, 1, 3),
            dropout_p=0.0,
            is_causal=False,
            scale=1 / self.head_dim**0.5,
        ).permute(0, 2, 1, 3)
        output = output.flatten(-2)
        c, x = output.split([seqlen1, seqlen2], dim=1)
        c = self.w1o(c)
        x = self.w2o(x)

        return c, x


class MMDiTBlock(nn.Module):
    def __init__(self, dim, heads=8, global_conddim=1024, is_last=False):
        super().__init__()

        self.normC1 = nn.LayerNorm(dim, elementwise_affine=False, bias=False)
        self.normC2 = nn.LayerNorm(dim, elementwise_affine=False, bias=False)
        if not is_last:
            self.mlpC = MLP(dim, hidden_dim=dim * 4)
            self.modC = nn.Sequential(
                nn.SiLU(),
                nn.Linear(global_conddim, 6 * dim, bias=False),
            )
        else:
            self.modC = nn.Sequential(
                nn.SiLU(),
                nn.Linear(global_conddim, 2 * dim, bias=False),
            )

        self.normX1 = nn.LayerNorm(dim, elementwise_affine=False, bias=False)
        self.normX2 = nn.LayerNorm(dim, elementwise_affine=False, bias=False)
        self.mlpX = MLP(dim, hidden_dim=dim * 4)
        self.modX = nn.Sequential(
            nn.SiLU(),
            nn.Linear(global_conddim, 6 * dim, bias=False),
        )

        self.attn = CrossAttention(dim, heads)
        self.is_last = is_last

    @torch.compile()
    def forward(self, c, x, global_cond, **kwargs):
        cres, xres = c, x
        
        # cpath
        cshift_msa, cscale_msa, cgate_msa, cshift_mlp, cscale_mlp, cgate_mlp = (
            self.modC(global_cond).chunk(6, dim=1)
        )
        c = modulate(self.normC1(c), cshift_msa[:, None, :], cscale_msa[:, None, :])

        # xpath
        xshift_msa, xscale_msa, xgate_msa, xshift_mlp, xscale_mlp, xgate_mlp = (
            self.modX(global_cond).chunk(6, dim=1)
        )

        x = modulate(self.normX1(x), xshift_msa[:, None, :], xscale_msa[:, None, :])

        # attention
        c, x = self.attn(c, x)

        c = self.normC2(cres + cgate_msa.unsqueeze(1) * c)
        c = cgate_mlp.unsqueeze(1) * self.mlpC(modulate(c, cshift_mlp[:, None, :], cscale_mlp[:, None, :]))
        c = cres + c

        x = self.normX2(xres + xgate_msa.unsqueeze(1) * x)
        x = xgate_mlp.unsqueeze(1) * self.mlpX(modulate(x, xshift_mlp[:, None, :], xscale_mlp[:, None, :]))
        x = xres + x

        return c, x
    


class PatchEmbed(nn.Module):
    def __init__(self, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.patch_proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size,
            bias=True
        )
        self.patch_size = patch_size

    @torch.compile()
    def forward(self, x):
        B, C, H, W = x.shape
        x = self.patch_proj(x)
        x = rearrange(x, "b c h w -> b (h w) c")
        return x
    
    def unpatchify(self, x, h, w):
        x = rearrange(
            x,
            "b (h w) (p1 p2 c) -> b c (h p1) (w p2)",
            h=h // self.patch_size,
            w=w // self.patch_size,
            p1=self.patch_size,
            p2=self.patch_size,
        )
        return x
    

class FinalLayer(nn.Module):
    """
    The final layer of MMDiT.
    """
    def __init__(
        self,
        hidden_size: int,
        global_conddim: int,
        patch_size: int,
        out_channels: int,
        total_out_channels: None|int = None,
        dtype=None,
        device=None,
    ):
        super().__init__()
        self.norm_final = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6, #dtype=dtype, device=device
        )
        self.linear = (
            nn.Linear(
                hidden_size,
                patch_size * patch_size * out_channels,
                bias=True,
                dtype=dtype,
                device=device,
            )
            if (total_out_channels is None)
            else nn.Linear(
                hidden_size, total_out_channels, bias=False, #dtype=dtype, device=device
            )
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                global_conddim, 2 * hidden_size, bias=False, #dtype=dtype, device=device
            ),
        )
        # nn.init.zeros_(self.adaLN_modulation.weight)
        # nn.init.zeros_(self.adaLN_modulation.bias)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    @torch.compile
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(t).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift[:, None, :], scale[:, None, :])
        x = self.linear(x)
        return x


class MMDiT(nn.Module):
    def __init__(
        self,
        in_channels: int=16,
        patch_size: int=2,
        depth: int=8,
        num_heads: int=4,
        mlp_ratio: float=4.0,
        context_input_size: int=1024,
        hidden_size: int=128,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.depth = depth
        self.mlp_ratio = mlp_ratio
        self.n_double_layers = depth // 2
        self.n_single_layers = depth - self.n_double_layers

        # Parse Layers
        self.t_embedder = TimestepEmbedder(context_input_size)
        # self.y_embedder = VectorEmbedder(context_input_size, context_input_size)
        self.patch_embed = PatchEmbed(patch_size, in_channels, hidden_size)
        self.context_seq_linear = nn.Linear(
            context_input_size, hidden_size, bias=False
        )
        self.positional_embedding = nn.Parameter(torch.randn(1, 1024, hidden_size) * 0.1)

        # Transformer Blocks
        self.double_layers = nn.ModuleList([])
        self.single_layers = nn.ModuleList([])

        for idx in range(self.n_single_layers):
            self.single_layers.append(
                DiTBlock(hidden_size, num_heads, context_input_size)
            )
        for idx in range(self.n_double_layers):
            self.double_layers.append(
                MMDiTBlock(
                    hidden_size, num_heads, context_input_size, 
                    # is_last=(idx == self.n_double_layers - 1)
                    is_last=False
                )
            )        

        # Final Layer
        self.final_layer = FinalLayer(
            hidden_size=hidden_size, 
            global_conddim=context_input_size, 
            patch_size=patch_size, 
            out_channels=in_channels
        )

        # Init weights
        for pn, p in self.named_parameters():
            if ".mod" in pn:
                nn.init.constant_(p, 0)
                # print("zeroed", pn)
        nn.init.constant_(self.context_seq_linear.weight, 0)
        # for layer in self.y_embedder.mlp:
        #     if hasattr(layer, 'weight'):
        #         nn.init.constant_(layer.weight, 0)
        #     if hasattr(layer, 'bias'):
        #         nn.init.constant_(layer.bias, 0)

    def forward(self, x, ctx, t, **kwargs):
        b, c, h, w = x.shape
        x = self.patch_embed(x)
        x = x + self.positional_embedding.repeat(b, 1, 1)[:, : x.shape[1], :]
        c_emb = self.context_seq_linear(ctx)
        # y_emb = self.y_embedder(ctx.mean(dim=1))
        y_emb = ctx.mean(dim=1)
        t_emb = self.t_embedder(t)
        t_emb = t_emb + y_emb

        for layer in self.double_layers:
            c, x = layer(c_emb, x, t_emb, **kwargs)

        c_len = c.size(1)
        cx = torch.cat([c, x], dim=1)
        for layer in self.single_layers:
            cx = layer(cx, t_emb, **kwargs)

        x = cx[:, c_len:]
        x = self.final_layer(x, t_emb)
        x = self.patch_embed.unpatchify(x, h, w)

        return x



def test_model():
    device = "cuda:0"
    dtype = torch.bfloat16

    context_hidden_size = 1024
    model = MMDiT(
        in_channels=16,
        patch_size=2,
        hidden_size=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        context_input_size=context_hidden_size,
    )

    model.to(device, dtype=dtype)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count / 1e6 :.4f} M")

    bs = 64
    image_data = torch.randn(bs, 16, 32, 32).to(device, dtype=dtype)
    context = torch.randn(bs, 128, context_hidden_size).to(device, dtype=dtype)
    timesteps = torch.randint(0, 1000, (bs,)).to(device, dtype=dtype)

    output = model(image_data, context, timesteps)
    for _ in tqdm(range(100)):
        output = model(image_data, context, timesteps)
    print(f"Output shape: {output.shape}")
    assert (
    image_data.shape == output.shape
    ), f"Input shape: {image_data.shape}, Output shape: {output.shape}"




if __name__ == "__main__":
    print('Testing -> model')
    test_model()
    