
# Code based on:
# https://github.com/Alpha-VLLM/LLaMA2-Accessory/blob/main/Large-DiT-T2I/models/model.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

#############################################


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=False),
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
        freqs = 1000 * torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half) / half
        ).to(t.device)
        args = t[:, None] * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    @torch.compile()
    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size).to(
            dtype=next(self.parameters()).dtype
        )
        t_emb = self.mlp(t_freq)
        return t_emb
    

class MultiHeadLayerNorm(nn.Module):
    def __init__(self, hidden_size=None, eps=1e-5):
        # Copy pasta from https://github.com/huggingface/transformers/blob/e5f71ecaae50ea476d1e12351003790273c4b2ed/src/transformers/models/cohere/modeling_cohere.py#L78

        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size)) #type: ignore
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        device = hidden_states.device

        hidden_states = hidden_states.to(torch.float32)
        mean = hidden_states.mean(-1, keepdim=True)
        variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
        hidden_states = (hidden_states - mean) * torch.rsqrt(
            variance + self.variance_epsilon
        )
        hidden_states = self.weight.to(torch.float32) * hidden_states
        return hidden_states.to(input_dtype)
    


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
    


class Fp32LayerNorm(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input):
        output = F.layer_norm(
            input.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        return output.type_as(input)



# class Attention(nn.Module):
#     def __init__(self, dim, n_heads):
#         super().__init__()

#         self.n_heads = n_heads
#         self.n_rep = 1
#         self.head_dim = dim // n_heads

#         self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)
#         self.wk = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
#         self.wv = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
#         self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)

#         self.q_norm = nn.LayerNorm(self.n_heads * self.head_dim)
#         self.k_norm = nn.LayerNorm(self.n_heads * self.head_dim)

#     @staticmethod
#     def reshape_for_broadcast(freqs_cis, x):
#         ndim = x.ndim
#         assert 0 <= 1 < ndim
#         # assert freqs_cis.shape == (x.shape[1], x.shape[-1])
#         _freqs_cis = freqs_cis[: x.shape[1]]
#         shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
#         return _freqs_cis.view(*shape)

#     @staticmethod
#     def apply_rotary_emb(xq, xk, freqs_cis):
#         xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
#         xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
#         freqs_cis_xq = Attention.reshape_for_broadcast(freqs_cis, xq_)
#         freqs_cis_xk = Attention.reshape_for_broadcast(freqs_cis, xk_)

#         xq_out = torch.view_as_real(xq_ * freqs_cis_xq).flatten(3)
#         xk_out = torch.view_as_real(xk_ * freqs_cis_xk).flatten(3)
#         return xq_out, xk_out

#     def forward(self, x, freqs_cis):
#         bsz, seqlen, _ = x.shape

#         xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

#         dtype = xq.dtype

#         xq = self.q_norm(xq)
#         xk = self.k_norm(xk)

#         xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
#         xk = xk.view(bsz, seqlen, self.n_heads, self.head_dim)
#         xv = xv.view(bsz, seqlen, self.n_heads, self.head_dim)

#         xq, xk = self.apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
#         xq, xk = xq.to(dtype), xk.to(dtype)

#         output = F.scaled_dot_product_attention(
#             xq.permute(0, 2, 1, 3),
#             xk.permute(0, 2, 1, 3),
#             xv.permute(0, 2, 1, 3),
#             dropout_p=0.0,
#             is_causal=False,
#         ).permute(0, 2, 1, 3)
#         output = output.flatten(-2)

#         return self.wo(output)
    


class DoubleAttention(nn.Module):
    def __init__(self, dim, n_heads, mh_qknorm=False):
        super().__init__()

        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        # this is for cond
        self.w1q = nn.Linear(dim, dim, bias=False)
        self.w1k = nn.Linear(dim, dim, bias=False)
        self.w1v = nn.Linear(dim, dim, bias=False)
        self.w1o = nn.Linear(dim, dim, bias=False)

        # this is for x
        self.w2q = nn.Linear(dim, dim, bias=False)
        self.w2k = nn.Linear(dim, dim, bias=False)
        self.w2v = nn.Linear(dim, dim, bias=False)
        self.w2o = nn.Linear(dim, dim, bias=False)

        self.q_norm1 = (
            MultiHeadLayerNorm((self.n_heads, self.head_dim))
            if mh_qknorm
            else RMSNorm(self.head_dim) #Fp32LayerNorm(self.head_dim, bias=False, elementwise_affine=False)
        )
        self.k_norm1 = (
            MultiHeadLayerNorm((self.n_heads, self.head_dim))
            if mh_qknorm
            else RMSNorm(self.head_dim) #Fp32LayerNorm(self.head_dim, bias=False, elementwise_affine=False)
        )

        self.q_norm2 = (
            MultiHeadLayerNorm((self.n_heads, self.head_dim))
            if mh_qknorm
            else RMSNorm(self.head_dim) #Fp32LayerNorm(self.head_dim, bias=False, elementwise_affine=False)
        )
        self.k_norm2 = (
            MultiHeadLayerNorm((self.n_heads, self.head_dim))
            if mh_qknorm
            else RMSNorm(self.head_dim) #Fp32LayerNorm(self.head_dim, bias=False, elementwise_affine=False)
        )


    @staticmethod
    def reshape_for_broadcast(freqs_cis, x):
        ndim = x.ndim
        assert 0 <= 1 < ndim
        # assert freqs_cis.shape == (x.shape[1], x.shape[-1])
        _freqs_cis = freqs_cis[: x.shape[1]]
        shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return _freqs_cis.view(*shape)

    @staticmethod
    def apply_rotary_emb(xq, xk, freqs_cis):
        xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
        xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
        freqs_cis_xq = DoubleAttention.reshape_for_broadcast(freqs_cis, xq_)
        freqs_cis_xk = DoubleAttention.reshape_for_broadcast(freqs_cis, xk_)

        xq_out = torch.view_as_real(xq_ * freqs_cis_xq).flatten(3)
        xk_out = torch.view_as_real(xk_ * freqs_cis_xk).flatten(3)
        return xq_out, xk_out
    

    @torch.compile()
    def forward(self, c, x):
        bsz, seqlen1, _ = c.shape
        bsz, seqlen2, _ = x.shape
        seqlen = seqlen1 + seqlen2

        cq, ck, cv = self.w1q(c), self.w1k(c), self.w1v(c)
        cq = cq.view(bsz, seqlen1, self.n_heads, self.head_dim)
        ck = ck.view(bsz, seqlen1, self.n_heads, self.head_dim)
        cv = cv.view(bsz, seqlen1, self.n_heads, self.head_dim)
        cq, ck = self.q_norm1(cq), self.k_norm1(ck)

        xq, xk, xv = self.w2q(x), self.w2k(x), self.w2v(x)
        xq = xq.view(bsz, seqlen2, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen2, self.n_heads, self.head_dim)
        xv = xv.view(bsz, seqlen2, self.n_heads, self.head_dim)
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
    


def find_multiple(n: int, k: int) -> int:
    if n % k == 0:
        return n
    return n + k - (n % k)



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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.c_fc1(x)) * self.c_fc2(x)
        x = self.c_proj(x)
        return x



class TransformerBlock(nn.Module):
    def __init__(self, dim, heads=8, global_conddim=1024, is_last=False):
        super().__init__()

        self.normC1 = RMSNorm(dim) #Fp32LayerNorm(dim, elementwise_affine=False, bias=False)
        self.normC2 = RMSNorm(dim) #Fp32LayerNorm(dim, elementwise_affine=False, bias=False)
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

        self.normX1 = RMSNorm(dim) #Fp32LayerNorm(dim, elementwise_affine=False, bias=False)
        self.normX2 = RMSNorm(dim) #Fp32LayerNorm(dim, elementwise_affine=False, bias=False)
        self.mlpX = MLP(dim, hidden_dim=dim * 4)
        self.modX = nn.Sequential(
            nn.SiLU(),
            nn.Linear(global_conddim, 6 * dim, bias=False),
        )

        self.attn = DoubleAttention(dim, heads)
        self.is_last = is_last

    @torch.compile()
    def forward(self, c, x, global_cond, **kwargs):
        cres, xres = c, x
        
        cshift_msa, cscale_msa, cgate_msa, cshift_mlp, cscale_mlp, cgate_mlp = (
            self.modC(global_cond).chunk(6, dim=1)
        )
    
        c = modulate(self.normC1(c), cshift_msa, cscale_msa)

        # xpath
        xshift_msa, xscale_msa, xgate_msa, xshift_mlp, xscale_mlp, xgate_mlp = (
            self.modX(global_cond).chunk(6, dim=1)
        )
        x = modulate(self.normX1(x), xshift_msa, xscale_msa)

        # attention
        c, x = self.attn(c, x)


        c = self.normC2(cres + cgate_msa.unsqueeze(1) * c)
        c = cgate_mlp.unsqueeze(1) * self.mlpC(modulate(c, cshift_mlp, cscale_mlp))
        c = cres + c

        x = self.normX2(xres + xgate_msa.unsqueeze(1) * x)
        x = xgate_mlp.unsqueeze(1) * self.mlpX(modulate(x, xshift_mlp, xscale_mlp))
        x = xres + x

        return c, x


# class FinalLayer(nn.Module):
#     def __init__(self, hidden_size, patch_size, out_channels):
#         super().__init__()
#         self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
#         self.linear = nn.Linear(
#             hidden_size, patch_size * patch_size * out_channels, bias=False
#         )
#         self.adaLN_modulation = nn.Sequential(
#             nn.SiLU(),
#             nn.Linear(min(hidden_size, 1024), 2 * hidden_size, bias=False),
#         )
#         # # init zero
#         nn.init.constant_(self.linear.weight, 0)

#     def forward(self, x, c):
#         shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
#         x = modulate(self.norm_final(x), shift, scale)
#         x = self.linear(x)
#         return x


class DiT_Llama(nn.Module):
    def __init__(
        self,
        in_channels=3,
        input_size=32,
        patch_size=2,
        dim=512,
        n_layers=5,
        n_heads=16,
        max_seq=16 * 16,
        cap_feat_dim=768
    ):
        super().__init__()
        self.global_cond_dim = min(dim, 1024)
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.input_size = input_size
        self.patch_size = patch_size

        self.init_x_linear = nn.Linear(
            patch_size * patch_size * in_channels, dim, 
            bias=False
        )

        self.x_embedder = nn.Linear(
            patch_size * patch_size * dim // 2, dim, 
            bias=False
        )
        nn.init.constant_(self.x_embedder.weight, 0)

        self.t_embedder = TimestepEmbedder(self.global_cond_dim)
        self.cap_embedder = nn.Linear(
            cap_feat_dim, self.global_cond_dim, 
            bias=False,
        )
        nn.init.constant_(self.cap_embedder.weight, 0)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    dim,
                    n_heads,
                    global_conddim=dim,
                    is_last=False#(layer_id == n_layers - 1),
                )
                for layer_id in range(n_layers)
            ]
        )
        self.final_linear = nn.Linear(
            dim, patch_size * patch_size * in_channels, bias=False
        )
        self.positional_encoding = nn.Parameter(torch.randn(1, max_seq, dim) * 0.1)
        self.h_max = int(max_seq**0.5)
        self.w_max = int(max_seq**0.5)

        for pn, p in self.named_parameters():
            if ".mod" in pn:
                nn.init.constant_(p, 0)
            if p.requires_grad:
                print(f"{pn} - {p.numel()}")


    @torch.no_grad()
    def extend_pe(self, init_dim=(16, 16), target_dim=(64, 64)):
        # extend pe
        pe_data = self.positional_encoding.data.squeeze(0)[: init_dim[0] * init_dim[1]]
        pe_as_2d = pe_data.view(init_dim[0], init_dim[1], -1).permute(2, 0, 1)

        # now we need to extend this to target_dim. for this we will use interpolation.
        # we will use torch.nn.functional.interpolate
        pe_as_2d = F.interpolate(
            pe_as_2d.unsqueeze(0), size=target_dim, mode="bilinear"
        )
        pe_new = pe_as_2d.squeeze(0).permute(1, 2, 0).flatten(0, 1)
        self.positional_encoding.data = pe_new.unsqueeze(0).contiguous()
        self.h_max, self.w_max = target_dim
        print("PE extended to", target_dim)

    def pe_selection_index_based_on_dim(self, h, w):
        h_p, w_p = h // self.patch_size, w // self.patch_size
        original_pe_indexes = torch.arange(self.positional_encoding.shape[1])
        original_pe_indexes = original_pe_indexes.view(self.h_max, self.w_max)
        original_pe_indexes = original_pe_indexes[
            self.h_max // 2 - h_p // 2 : self.h_max // 2 + h_p // 2,
            self.w_max // 2 - w_p // 2 : self.w_max // 2 + w_p // 2,
        ]
        return original_pe_indexes.flatten()
    

    def unpatchify(self, x, h, w):
        c = self.out_channels
        p = self.patch_size

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs


    def patchify(self, x):
        B, C, H, W = x.size()
        x = x.view(
            B,
            C,
            H // self.patch_size,
            self.patch_size,
            W // self.patch_size,
            self.patch_size,
        )
        x = x.permute(0, 2, 4, 1, 3, 5).flatten(-3).flatten(1, 2)
        return x

    def forward(self, x, t, caption):
        b, c, h, w = x.shape
        # w conv -> torch.Size([2, 256, 768])
        # wo conv -> torch.Size([2, 256, 64])
        x = self.init_x_linear(self.patchify(x)) # B, T_x, D
        x = x + self.positional_encoding[:, :x.size(1)]

        t = self.t_embedder(t)  # (N, D)
        cap = self.cap_embedder(caption)  # (N, T, D)
        cap_pool = cap.mean(dim=1)
        global_cond = t + cap_pool
        for idx, layer in enumerate(self.layers):
            cap, x = layer(cap, x, global_cond)

        x = self.final_linear(x)
        x = self.unpatchify(x, h // self.patch_size, w // self.patch_size)  # (N, out_channels, H, W)
        return x

    def forward_with_cfg(self, x, t, caption, cfg_scale):
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        caption = torch.cat([caption, caption], dim=0)
        model_out = self.forward(combined, t, caption)
        eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)

    @staticmethod
    def precompute_freqs_cis(dim, end, theta=10000.0):
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        t = torch.arange(end)
        freqs = torch.outer(t, freqs).float()
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        return freqs_cis


def DiT_Llama_S(**kwargs):
    return DiT_Llama(in_channels=16, patch_size=2, dim=384, n_layers=12, n_heads=6, **kwargs)

def DiT_Llama_B(**kwargs):
    return DiT_Llama(in_channels=16, patch_size=2, dim=528, n_layers=12, n_heads=12, **kwargs)

if __name__ == "__main__":
    model = DiT_Llama_S()
    model.eval()
    print(f"Num parameters: {sum(p.numel() for p in model.parameters())}")
    x = torch.randn(2, 4, 32, 32)
    t = torch.randint(0, 1000, (2,))
    cap = torch.randn(2, 128, 768)

    with torch.no_grad():
        out = model(x, t, cap)
        print(out.shape)