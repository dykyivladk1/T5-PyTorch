import torch
import torch.nn as nn

from utils import *

from dataclasses import dataclass



@dataclass
class Config:
    vocab_size: int
    layers: int
    model_dim: int
    ff_dim: int
    heads: int
    head_dim: int
    rel_pos_buckets: int
    eps: float = 1e-6





class EmbedLayer(nn.Module):
    def __init__(self, cfg: Config):
        super(EmbedLayer, self).__init__()
        self.word_emb = nn.Embedding(cfg.vocab_size, cfg.model_dim)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.word_emb(x)






class PosEmbed(nn.Module):
    def __init__(self, cfg: Config):
        super(PosEmbed, self).__init__()
        self.pos_emb = nn.Embedding(cfg.rel_pos_buckets, cfg.heads)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pos_emb(x)
    



class Norm(nn.Module):
    def __init__(self, cfg: Config, eps=1e-6):
        super(Norm, self).__init__()
        self.weight = nn.Parameter(torch.ones(cfg.model_dim))
        self.eps = eps
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(var * self.eps)
        return self.weight * x




class MLP(nn.Module):
    def __init__(self, cfg: Config):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(cfg.model_dim, cfg.ff_dim, bias=False)
        self.fc2 = nn.Linear(cfg.ff_dim, cfg.model_dim, bias=False)
        self.act = nn.ReLU()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.fc1(x))
        return self.fc2(x)
    




class SelfAttn(nn.Module):
    def __init__(self, cfg: Config, decoder: bool):
        super(SelfAttn, self).__init__()
        self.model_dim = cfg.model_dim
        self.heads = cfg.heads
        self.head_dim = cfg.head_dim
        self.q = nn.Linear(cfg.model_dim, cfg.model_dim, bias=False)
        self.k = nn.Linear(cfg.model_dim, cfg.model_dim, bias=False)
        self.v = nn.Linear(cfg.model_dim, cfg.model_dim, bias=False)
        self.out = nn.Linear(cfg.model_dim, cfg.model_dim, bias=False)
        self.decoder = decoder
    def split(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.size()
        x = x.view(b, n, self.heads, self.head_dim).transpose(1, 2)
        return x
    def unify(self, x: torch.Tensor) -> torch.Tensor:
        b, _, n, _ = x.size()
        x = x.transpose(1, 2).contiguous().view(b, n, self.model_dim)
        return x
    def forward(self, x: torch.Tensor, pos_bias: torch.Tensor) -> torch.Tensor:
        if self.decoder:
            Q = self.q(x[:, -1:, :])
        else:
            Q = self.q(x)
        K = self.k(x)
        V = self.v(x)
        Q, K, V = self.split(Q), self.split(K), self.split(V)
        scores = Q @ K.transpose(-2, -1)
        if self.decoder:
            pos_bias = pos_bias[:, :, -1:, :]
        scores += pos_bias
        attn = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        attn_out = attn @ V
        attn_out = self.unify(attn_out)
        return self.out(attn_out)


    

class EncDecAttn(nn.Module):
    def __init__(self, cfg: Config):
        super(EncDecAttn, self).__init__()
        self.model_dim = cfg.model_dim
        self.heads = cfg.heads
        self.head_dim = cfg.head_dim
        self.q = nn.Linear(cfg.model_dim, cfg.model_dim, bias=False)
        self.k = nn.Linear(cfg.model_dim, cfg.model_dim, bias=False)
        self.v = nn.Linear(cfg.model_dim, cfg.model_dim, bias=False)
        self.out = nn.Linear(cfg.model_dim, cfg.model_dim, bias=False)
    def split(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.size()
        x = x.view(b, n, self.heads, self.head_dim).transpose(1, 2)
        return x
    def unify(self, x: torch.Tensor) -> torch.Tensor:
        b, _, n, _ = x.size()
        x = x.transpose(1, 2).contiguous().view(b, n, self.model_dim)
        return x
    def forward(self, x: torch.Tensor, enc: torch.Tensor, pos_bias: torch.Tensor) -> torch.Tensor:
        Q = self.q(x)
        K = self.k(enc)
        V = self.v(enc)
        Q, K, V = self.split(Q), self.split(K), self.split(V)
        scores = Q @ K.transpose(-2, -1)
        scores += pos_bias
        attn = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        attn_out = attn @ V
        attn_out = self.unify(attn_out)
        return self.out(attn_out)
    



cfg = Config(vocab_size=32128, layers=6, model_dim=512, ff_dim=2048, heads=8, head_dim=64, rel_pos_buckets=32)




class EncLayer(nn.Module):
    def __init__(self, cfg: Config):
        super(EncLayer, self).__init__()
        self.self_attn = SelfAttn(cfg, False)
        self.norm1 = Norm(cfg)
        self.mlp = MLP(cfg)
        self.norm2 = Norm(cfg)
    def forward(self, x: torch.Tensor, pos_bias: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.norm1(x), pos_bias)
        x = x + self.mlp(self.norm2(x))
        return x


class DecLayer(nn.Module):
    def __init__(self, cfg: Config):
        super(DecLayer, self).__init__()
        self.self_attn = SelfAttn(cfg, True)
        self.norm1 = Norm(cfg)
        self.enc_dec_attn = EncDecAttn(cfg)
        self.norm2 = Norm(cfg)
        self.mlp = MLP(cfg)
        self.norm3 = Norm(cfg)
    def forward(self, x: torch.Tensor, enc: torch.Tensor, self_bias: torch.Tensor, enc_bias: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.norm1(x), self_bias)
        x = x + self.enc_dec_attn(self.norm2(x), enc, enc_bias)
        x = x + self.mlp(self.norm3(x))
        return x


class Encoder(nn.Module):
    def __init__(self, cfg: Config):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([EncLayer(cfg) for _ in range(cfg.layers)])
        self.pos_embed = PosEmbed(cfg)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pos_bias = bias(x.size(1), x.size(1), self.pos_embed, False)
        for layer in self.layers:
            x = layer(x, pos_bias)
        return x


class Decoder(nn.Module):
    def __init__(self, cfg: Config):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([DecLayer(cfg) for _ in range(cfg.layers)])
        self.self_pos = PosEmbed(cfg)
        self.enc_dec_pos = PosEmbed(cfg)
    def forward(self, x: torch.Tensor, enc: torch.Tensor) -> torch.Tensor:
        self_bias = bias(1, x.size(1), self.self_pos, True)
        enc_bias = bias(1, enc.size(1), self.enc_dec_pos, True)
        for layer in self.layers:
            x = layer(x, enc, self_bias, enc_bias)
        return x[:, -1:, :]




class Transformer(nn.Module):
    def __init__(self, cfg: Config):
        super(Transformer, self).__init__()
        self.embed = EmbedLayer(cfg)
        self.enc = Encoder(cfg)
        self.enc_norm = Norm(cfg)
        self.dec = Decoder(cfg)
        self.dec_norm = Norm(cfg)
        self.head = nn.Linear(cfg.model_dim, cfg.vocab_size, bias=False)
        self.model_dim = cfg.model_dim
    @torch.no_grad()
    def forward(self, x: torch.Tensor, max_toks=50, dbg=False) -> torch.Tensor:
        emb = self.embed(x)
        enc_out = self.enc(emb)
        enc_out = self.enc_norm(enc_out)
        out = torch.tensor([[0]], device=x.device)
        while out[:, -1] != 1 and len(out[0]) < max_toks:
            if dbg:
                print(f"Current outputs: {out}")
            dec_emb = self.embed(out)
            dec_out = self.dec(dec_emb, enc_out)
            dec_out = self.dec_norm(dec_out)
            dec_out *= (self.model_dim ** -0.5)
            logits = self.head(dec_out[:, -1, :])
            topk = 10
            probs, idx = torch.topk(logits, k=topk, dim=-1)
            probs = torch.softmax(probs, dim=-1)
            tok = torch.multinomial(probs, 1)
            tok = idx.gather(-1, tok)
            out = torch.cat([out, tok], dim=-1)
        return out

