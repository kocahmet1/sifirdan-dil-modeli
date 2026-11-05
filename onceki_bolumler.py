"""Sıfırdan Dil Modeli · önceki bölümlerin kodu.

Her defter bir öncekinin bittiği yerden başlar. Bu dosya önceki bölümlerde yazdığımız kodu tek yerde toplar
(kitabın deposundaki previous_chapters.py'nin karşılığı). Bir defterin ilk hücresi onu indirir ve içinden alır:

    from onceki_bolumler import load_tokenizer, create_dataloader_v1

Bölüm 2 · BPE: kelimeyi parçalara bölmek    →  merge, BPETokenizer, load_tokenizer (Bölüm 3'ün defterindeki adıyla: cirak_tokenizer)
Bölüm 3 · Kayan pencere: girdi ve hedef     →  GPTDatasetV1, create_dataloader_v1
Bölüm 5 · Dikkat I: basit öz-dikkat         →  isi_haritasi
Bölüm 6 · Dikkat II: sorgu, anahtar, değer  →  SelfAttention_v1, SelfAttention_v2
Bölüm 7 · Dikkat III: maske ve çok başlı    →  MultiHeadAttention
Bölüm 8 · GPT'nin parçaları                 →  GPT_CONFIG (GPT_CONFIG_CIRAK da aynı sözlük), LayerNorm, GELU, FeedForward

Kod, Sebastian Raschka'nın "Build a Large Language Model (From Scratch)" kitabının deposundaki kodun
(https://github.com/rasbt/LLMs-from-scratch, Apache License 2.0) Türkçe metin için uyarlamasıdır.
Değişiklikler: GPT-2'nin tokenizer'ı (tiktoken) yerine serinin kendi bayt düzeyli BPE'si (Çırak'ın tokenizer'ı,
4.096 parça); kitaptaki GPT_CONFIG_124M yerine Çırak'ın boyunda GPT_CONFIG (4.096 parça, bağlam 256, 256 boyut,
8 dikkat başlığı, 6 blok, qkv_bias=True); Türkçe yorumlar.
"""

import json
import os
import re
import urllib.request

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------------------------------- Bölüm 2 · BPE
# Önce kelimelere ayır: önünde en fazla bir boşluk olan harf dizileri, rakam dizileri, noktalama, boşluklar.
PAT = re.compile(r" ?[^\W\d_]+| ?\d+| ?[^\s\w]+|\s+(?!\S)|\s+")


def merge(ids, pair, new_id):
    """Listede çiftin geçtiği her yeri tek bir yeni numarayla değiştirir."""
    out = []
    i = 0
    while i < len(ids):
        if i + 1 < len(ids) and (ids[i], ids[i + 1]) == pair:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class BPETokenizer:
    def __init__(self, merges):
        self.merges = merges
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.vocab[256] = b"<|endoftext|>"
        for (a, b), new_id in merges.items():
            self.vocab[new_id] = self.vocab[a] + self.vocab[b]

    def encode(self, text):
        out = []
        for word in PAT.findall(text):
            ids = list(word.encode("utf-8"))
            while len(ids) >= 2:
                pairs = [p for p in zip(ids, ids[1:]) if p in self.merges]
                if not pairs:
                    break
                pair = min(pairs, key=self.merges.get)
                ids = merge(ids, pair, self.merges[pair])
            out.extend(ids)
        return out

    def decode(self, ids):
        data = b"".join(self.vocab[i] for i in ids)
        return data.decode("utf-8", errors="replace")


def load_tokenizer(file_path="cirak_tokenizer.json"):
    """Bölüm 2'nin sonundaki gibi: Çırak'ın birleştirmelerini okuyup BPETokenizer'ı kurar (4.096 parça)."""
    with open(file_path, encoding="utf-8") as f:
        data = json.load(f)
    merges = {tuple(p): 257 + i for i, p in enumerate(data["merges"])}
    return BPETokenizer(merges)


REPO = "https://raw.githubusercontent.com/kocahmet1/sifirdan-dil-modeli/main/"


def cirak_tokenizer(file_path="cirak_tokenizer.json"):
    """Bölüm 3'ün defteri bu adla çağırır: load_tokenizer'ın aynısı; dosya yoksa önce depodan indirir."""
    if not os.path.exists(file_path):
        urllib.request.urlretrieve(REPO + "veri/cirak_tokenizer.json", file_path)
    return load_tokenizer(file_path)


# ---------------------------------------------------------------------------------------------- Bölüm 3 · kayan pencere
class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Bütün metni token numaralarına çevir
        token_ids = tokenizer.encode(txt)

        # Kayan pencere: max_length uzunluğunda parçalar, stride kadar kayarak
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]   # hedef: bir sağa kaydırılmış hâli
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True,
                         num_workers=0):
    # Kitapta burada GPT-2'nin tokenizer'ı yüklenir; bizde Çırak'ınki (Bölüm 2)
    tokenizer = load_tokenizer()

    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )
    return dataloader


# ---------------------------------------------------------------------------------------------- Bölüm 5 · basit öz-dikkat
def isi_haritasi(agirliklar, etiketler):
    """Bir dikkat ağırlık matrisini ısı haritası olarak çizer: satırlar bakan, sütunlar bakılan kelime."""
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 5))
    plt.imshow(agirliklar.detach(), cmap="viridis")
    plt.xticks(range(len(etiketler)), etiketler, rotation=45)
    plt.yticks(range(len(etiketler)), etiketler)
    plt.colorbar()
    plt.show()


# ---------------------------------------------------------------------------------------------- Bölüm 6 · sorgu, anahtar, değer
class SelfAttention_v1(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x):
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value
        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        context_vec = attn_weights @ values
        return context_vec


class SelfAttention_v2(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        context_vec = attn_weights @ values
        return context_vec


# ---------------------------------------------------------------------------------------------- Bölüm 7 · çok başlı dikkat
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads == 0), "d_out, num_heads'e tam bölünmeli"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads      # her başlığın boyu

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # başlıkların çıktısını birleştirir
        self.dropout = nn.Dropout(dropout)
        # üçgen maske: her token yalnız kendisine ve solundakilere bakar
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x)                    # şekil: (b, num_tokens, d_out)
        queries = self.W_query(x)
        values = self.W_value(x)

        # son boyutu başlıklara böl: (b, num_tokens, d_out) → (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # (b, num_tokens, num_heads, head_dim) → (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # her başlık için ölçekli nokta çarpım dikkati, maskeyle
        attn_scores = queries @ keys.transpose(2, 3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)

        # başlıkları birleştir: self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)

        return context_vec


# ---------------------------------------------------------------------------------------------- Bölüm 8 · GPT'nin parçaları
# Kitaptaki GPT_CONFIG_124M'nin (GPT-2'nin en küçüğü) yerine Çırak'ın boyu: Bölüm 13'te Çırak'ın eğitilmiş
# ağırlıkları bu ayarlarla kurulan modele birebir oturur.
GPT_CONFIG = {
    "vocab_size": 4096,      # sözlük: Çırak'ın BPE'si
    "context_length": 256,   # bağlam: en fazla 256 token
    "emb_dim": 256,          # her tokenın vektörü
    "n_heads": 8,            # dikkat başlığı
    "n_layers": 6,           # transformer bloğu
    "drop_rate": 0.1,        # dropout oranı
    "qkv_bias": True         # sorgu, anahtar, değer katmanlarında bias
}
GPT_CONFIG_CIRAK = GPT_CONFIG   # Bölüm 8'in defterindeki ad


class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)
