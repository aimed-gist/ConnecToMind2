"""
ConnecToMind2 - Model with Bottleneck MLP (BLIP-2 ITC+ITM Style)

models.py와 동일하지만, RegionLevelEmbedding과 ConnectomeQFormer 사이에
Bottleneck MLP (768 -> 128 -> 768)를 추가한 버전입니다.

Bottleneck MLP:
    - 768 -> 128 -> 768 (파라미터: 197,376개)
    - 정보 압축을 통한 representation learning 효과
    - 768 -> 768 단일 레이어 대비 약 3배 적은 파라미터

Architecture (from diagram):
    fMRI [B, 100, (roi+padding)]
        -> (a) Region-level embedding -> [B, 100, 768]
        -> (NEW) Bottleneck MLP -> [B, 100, 768]  <-- 추가됨
        -> (b) Connectome-Q-former -> [B, 100, 768]
        -> Linear layer -> [B, 257, 768]
        -> L2 norm -> [B, 257, 768]
        -> Versatile Diffusion -> Reconstructed Image [B, 512, 512]
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.utils import spectral_norm

from transformers import CLIPVisionModel, CLIPVisionModelWithProjection
from diffusers import DiffusionPipeline
from diffusers.models.autoencoder_kl import Decoder


# ============================================================================
# CLIP Image Encoder (그림의 좌측 하단) - ViT-L/14
# ============================================================================

class CLIPImageEncoder(nn.Module):
    """
    CLIP ViT-L/14 이미지 인코더

    그림 설명:
        Image -> CLIP ViT-L/14 -> Last hidden [B, 257, 1024]
                               -> Linear layer + L2 norm -> [B, 257, 768]
    """
    def __init__(self, pretrained_model="openai/clip-vit-large-patch14", freeze=True):
        super().__init__()
        # Use CLIPVisionModelWithProjection (includes pretrained visual_projection layer)
        self.clip_model = CLIPVisionModelWithProjection.from_pretrained(pretrained_model)

        if freeze:
            for param in self.clip_model.parameters():
                param.requires_grad = False

    def forward(self, images):
        """
        Input: images [B, 3, 224, 224]
        Output: hidden_state [B, 257, 768] - L2 normalized
        """
        outputs = self.clip_model(images, output_hidden_states=True)
        last_hidden = outputs.last_hidden_state  # [B, 257, 1024]

        # Post layer norm
        last_hidden = self.clip_model.vision_model.post_layernorm(last_hidden)  # [B, 257, 1024]

        # Pretrained visual projection + L2 norm
        hidden_state = self.clip_model.visual_projection(last_hidden)  # [B, 257, 768]
        # FP32로 L2 normalize (FP16 수치 불안정 방지)
        hidden_state = F.normalize(hidden_state.float(), dim=-1).to(hidden_state.dtype)

        return hidden_state


# ============================================================================
# Region-level Embedding (그림의 (a))
# ============================================================================

class RegionLevelEmbedding(nn.Module):
    """
    (a) Region-level embedding

    그림 설명:
        task-fMRI [B, 100, (roi+padding)]
        -> Flatten -> Linear projection -> [B, 100, 768]
    """
    def __init__(self, seq_len=100, input_dim=3291, embed_dim=768):
        super().__init__()
        self.embed_dim = embed_dim
        self.seq_len = seq_len

        # Region-level embedding: ROI별로 다른 linear layer
        self.linear_weight = nn.Parameter(torch.empty(seq_len, input_dim, embed_dim))
        for t in range(seq_len):
            init.xavier_uniform_(self.linear_weight[t])

        self.layernorm = nn.LayerNorm(embed_dim)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        """
        Input: x [B, 100, input_dim]
        Output: x [B, 100, 768]
        """
        # Ensure dtype compatibility with linear_weight
        x = x.to(dtype=self.linear_weight.dtype)

        # Region-level embedding (각 ROI별 linear)
        x = torch.einsum("btd,tdh->bth", x, self.linear_weight)  # [B, 100, 768]
        x = self.layernorm(x)
        x = self.gelu(x)
        x = self.dropout(x)
        return x


# ============================================================================
# Bottleneck MLP (NEW - 768 -> 128 -> 768)
# ============================================================================

class BottleneckMLP(nn.Module):
    """
    Bottleneck MLP: 768 -> 128 -> 768

    RegionLevelEmbedding과 ConnectomeQFormer 사이에 삽입하여
    정보 압축 및 representation learning 효과를 줌

    파라미터 수:
        - 768 * 128 + 128 = 98,432 (첫 번째 레이어)
        - 128 * 768 + 768 = 98,944 (두 번째 레이어)
        - Total: 197,376개

    비교:
        - 768 * 768 + 768 = 590,592개 (단일 레이어)
        - Bottleneck이 약 3배 적음
    """
    def __init__(self, embed_dim=768, bottleneck_dim=128, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.bottleneck_dim = bottleneck_dim

        self.down_proj = nn.Linear(embed_dim, bottleneck_dim)
        self.up_proj = nn.Linear(bottleneck_dim, embed_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """
        Input: x [B, 100, 768]
        Output: x [B, 100, 768]
        """
        residual = x

        # Down projection: 768 -> 128
        x = self.down_proj(x)  # [B, 100, 128]
        x = self.activation(x)
        x = self.dropout(x)

        # Up projection: 128 -> 768
        x = self.up_proj(x)  # [B, 100, 768]

        # Residual connection + LayerNorm
        x = self.layernorm(x + residual)

        return x


# ============================================================================
# FC Transformer Encoder (BottleneckMLP 대체 옵션)
# ============================================================================

class FCTransformerEncoderLayer(nn.Module):
    """
    Self-attention에 FC prior를 더하는 Transformer Encoder Layer.
    ConnectomeQFormer의 FC prior 방식을 그대로 차용.
    """
    def __init__(self, d_model=768, nhead=12, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.nhead = nhead
        self.head_dim = d_model // nhead

        # Self-Attention (Q, K, V projection)
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)

        # Feed Forward
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.activation = nn.GELU()

        # Normalization and dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.ffn_dropout = nn.Dropout(dropout)

    def forward(self, x, fc_prior_mask=None, connectivity_type='FC'):
        """
        Input:
            x [B, seq_len, d_model]
            fc_prior_mask [B*nhead, seq_len, seq_len] or None
            connectivity_type: 'FC' or 'SC'
        Output:
            x [B, seq_len, d_model]
        """
        B, T, E = x.shape

        # Self-attention with connectivity prior
        residual = x
        x_norm = self.norm1(x)

        qkv = self.qkv_proj(x_norm)  # [B, T, 3E]
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.nhead, self.head_dim).transpose(1, 2)  # [B, H, T, D]
        k = k.view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.nhead, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [B, H, T, T]

        # Connectivity prior 적용
        if fc_prior_mask is not None:
            # fc_prior_mask: [B*H, T, T] -> [B, H, T, T]
            fc_bias = fc_prior_mask.view(B, self.nhead, T, T)
            if connectivity_type == 'SC':
                # SC: sign(attn) * sc_bias (대비 증폭)
                attn_scores = attn_scores + torch.sign(attn_scores) * fc_bias
            else:
                # FC: 단순 덧셈
                attn_scores = attn_scores + fc_bias

        # FP32로 softmax (FP16 overflow 방지)
        attn_weights = torch.softmax(attn_scores.float(), dim=-1).to(attn_scores.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)  # [B, H, T, D]
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, E)
        attn_output = self.out_proj(attn_output)

        x = residual + self.dropout1(attn_output)

        # Feed Forward
        residual = x
        x_norm = self.norm2(x)
        x_ffn = self.linear2(self.ffn_dropout(self.activation(self.linear1(x_norm))))
        x = residual + self.dropout2(x_ffn)

        return x


class FCTransformerEncoder(nn.Module):
    """
    ROI 간 self-attention Transformer Encoder (FC prior 포함).

    BottleneckMLP 대체 옵션으로, RegionLevelEmbedding 후에 사용.
    FC prior 로딩은 ConnectomeQFormer의 방식을 그대로 차용.

    Input:  [B, 100, 768]
    Output: [B, 100, 768]
    """
    def __init__(self, d_model=768, nhead=12, num_layers=4, dropout=0.1,
                 is_fc=False, subjects=None, fc_base_dir=None,
                 seq_len=100, roi_suffix='schaefer100', fc_prior_scale=1.0,
                 connectivity_type='FC'):
        super().__init__()
        self.is_fc = is_fc
        self.nhead = nhead
        self.seq_len = seq_len
        self.fc_prior_scale = fc_prior_scale
        self.connectivity_type = connectivity_type

        self.layers = nn.ModuleList([
            FCTransformerEncoderLayer(d_model, nhead, d_model * 4, dropout)
            for _ in range(num_layers)
        ])

        self.final_ln = nn.LayerNorm(d_model)

        # Connectivity prior (FC or SC)
        self.fc_priors = {}
        if is_fc and subjects is not None and fc_base_dir is not None:
            self._load_fc_priors(subjects, fc_base_dir, seq_len, roi_suffix)

    def _load_fc_priors(self, subjects, fc_base_dir, seq_len, roi_suffix):
        """FC/SC prior 로드 (FC: arctanh, SC: log1p)"""
        import numpy as np

        conn_tag = self.connectivity_type  # 'FC' or 'SC'

        for sub in subjects:
            #fc_path = f"{fc_base_dir}/{sub}/{sub}_{conn_tag}_{roi_suffix}.npy"
            fc_path = f"{fc_base_dir}/{sub}/{sub}_{conn_tag}_{roi_suffix}.npy"

            if not os.path.exists(fc_path):
                print(f"  [FCTransformerEncoder] {conn_tag} prior not found: {fc_path}")
                continue

            fc_prior = np.load(fc_path).astype(np.float32)

            assert fc_prior.shape[0] == fc_prior.shape[1] == seq_len, \
                f"{conn_tag} prior shape mismatch for {sub}: {fc_prior.shape} vs {seq_len}"

            np.fill_diagonal(fc_prior, 0)

            if conn_tag == 'SC':
                fc_prior_z = np.log1p(fc_prior)
            else:
                fc_prior = np.clip(fc_prior, -0.9999, 0.9999)
                fc_prior_z = np.arctanh(fc_prior)

            mask = torch.from_numpy(fc_prior_z)
            mask = mask.unsqueeze(0).repeat(self.nhead, 1, 1)  # [H, seq_len, seq_len]

            self.fc_priors[sub] = mask

        print(f"  [FCTransformerEncoder] {len(self.fc_priors)} {conn_tag} priors loaded (nhead={self.nhead})")

    def _build_fc_prior_mask(self, subject_names, device, dtype=None):
        """ConnectomeQFormer._build_fc_prior_mask와 동일한 방식"""
        masks = []
        for sub_name in subject_names:
            if sub_name in self.fc_priors:
                mask = self.fc_priors[sub_name].to(device=device, dtype=dtype if dtype else self.fc_priors[sub_name].dtype)
            else:
                mask = torch.zeros(self.nhead, self.seq_len, self.seq_len,
                                   dtype=dtype if dtype else torch.float32, device=device)
            masks.append(mask)

        fc_prior_mask = torch.cat(masks, dim=0)  # [B*H, seq_len, seq_len]
        fc_prior_mask = fc_prior_mask * self.fc_prior_scale

        return fc_prior_mask

    def forward(self, x, subject_names=None):
        """
        Input:
            x [B, seq_len, 768]
            subject_names: list of subject names [B]
        Output:
            x [B, seq_len, 768]
        """
        fc_prior_mask = None
        if self.is_fc and subject_names is not None:
            fc_prior_mask = self._build_fc_prior_mask(subject_names, x.device, dtype=x.dtype)

        for layer in self.layers:
            x = layer(x, fc_prior_mask=fc_prior_mask, connectivity_type=self.connectivity_type)

        x = self.final_ln(x)
        return x


# ============================================================================
# Connectome-Q-Former (그림의 (b))
# ============================================================================

class ConnectomeQFormerBlock(nn.Module):
    """
    Connectome-Q-Former 블록: Self-attention + Cross-attention (optional) + Feed forward
    """
    def __init__(self, hidden_size=768, num_heads=12, intermediate_size=3072,
                 dropout=0.1, layer_norm_eps=1e-6):
        super().__init__()

        # Self-Attention
        self.self_ln = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.self_attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        self.self_drop = nn.Dropout(dropout)

        # Cross-Attention (query tokens attend to fMRI) - Query만 LayerNorm (BLIP-2 방식)
        self.cross_ln = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.cross_attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        self.cross_drop = nn.Dropout(dropout)

        # Feed Forward
        self.ffn_ln = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)
        self.ffn_drop = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x, fmri_feats, attn_mask=None, do_cross=True, n_q=None,
                cross_attn_mask=None, connectivity_type='FC'):
        """
        Input:
            x [B, L, 768] - query tokens (+ optional CLIP hidden states)
            fmri_feats [B, 100, 768] - fMRI embeddings
            attn_mask: attention mask for Q-T separation
            do_cross: whether to do cross-attention in this layer
            n_q: number of query tokens (for extracting query part in cross-attention)
            cross_attn_mask: connectivity prior mask [B*H, n_q, seq_len]
            connectivity_type: 'FC' or 'SC'
        Output:
            x [B, L, 768]
        """
        # Self-Attention (with mask if provided)
        residual = x
        x_norm = self.self_ln(x)
        x_sa, _ = self.self_attn(x_norm, x_norm, x_norm,
                                  attn_mask=attn_mask, need_weights=False)
        x = residual + self.self_drop(x_sa)

        # Cross-Attention (query -> fMRI) - 조건부 실행, query 부분만
        if do_cross:
            q = x[:, :n_q, :]  #  Query 부분만 cross-attention 적용 -> [B, n_q, 768]
            q_res = q
            q_norm = self.cross_ln(q)

            if connectivity_type == 'SC' and cross_attn_mask is not None:
                # SC: manual cross-attention with sign(attn) * sc_bias
                q_ca = self._sc_cross_attention(q_norm, fmri_feats, cross_attn_mask)
            else:
                # FC: nn.MultiheadAttention with additive bias
                q_ca, _ = self.cross_attn(q_norm, fmri_feats, fmri_feats,
                                         attn_mask=cross_attn_mask,
                                         need_weights=False)

            q = q_res + self.cross_drop(q_ca)
            x = torch.cat([q, x[:, n_q:, :]], dim=1)  # Query + 나머지 다시 concat

        # Feed Forward
        residual = x
        x_norm = self.ffn_ln(x)
        x_ffn = self.fc2(self.ffn_drop(self.activation(self.fc1(x_norm))))
        x = residual + x_ffn

        return x

    def _sc_cross_attention(self, q_norm, fmri_feats, sc_mask):
        """SC prior용 manual cross-attention: sign(attn_scores) * sc_bias"""
        B = q_norm.size(0)
        E = self.cross_attn.embed_dim
        num_heads = self.cross_attn.num_heads
        head_dim = E // num_heads

        # Q, K, V projection (cross_attn의 weight 재사용)
        w = self.cross_attn.in_proj_weight
        b = self.cross_attn.in_proj_bias
        q_proj = F.linear(q_norm, w[:E], b[:E])
        k_proj = F.linear(fmri_feats, w[E:2*E], b[E:2*E])
        v_proj = F.linear(fmri_feats, w[2*E:], b[2*E:])

        # [B, T, E] -> [B, H, T, D]
        n_q = q_proj.size(1)
        n_k = k_proj.size(1)
        q_proj = q_proj.view(B, n_q, num_heads, head_dim).transpose(1, 2)
        k_proj = k_proj.view(B, n_k, num_heads, head_dim).transpose(1, 2)
        v_proj = v_proj.view(B, n_k, num_heads, head_dim).transpose(1, 2)

        # Attention scores
        attn_scores = torch.matmul(q_proj, k_proj.transpose(-2, -1)) / (head_dim ** 0.5)

        # SC prior: sign(attn) * sc_bias
        sc_bias = sc_mask.view(B, num_heads, n_q, n_k)
        attn_scores = attn_scores + torch.sign(attn_scores) * sc_bias

        # Softmax (FP32)
        attn_weights = torch.softmax(attn_scores.float(), dim=-1).to(attn_scores.dtype)

        # Output
        attn_output = torch.matmul(attn_weights, v_proj)  # [B, H, n_q, D]
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, n_q, E)
        attn_output = self.cross_attn.out_proj(attn_output)

        return attn_output


class ConnectomeQFormer(nn.Module):
    """
    (b) Connectome-Q-Former (initialized from CLIP ViT-L/14)
    """
    def __init__(self, hidden_size=768, num_heads=12, num_layers=12,
                 num_query_tokens=100, dropout=0.1, cross_attention_freq=2,
                 clip_model_name="openai/clip-vit-base-patch16",
                 is_fc=False, subjects=None, fc_base_dir=None, roi_suffix='schaefer100',
                 fc_prior_scale=1.0, connectivity_type='FC'):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_query_tokens = num_query_tokens
        self.num_layers = num_layers
        self.cross_attention_freq = cross_attention_freq
        self.is_fc = is_fc
        self.num_heads = num_heads
        self.connectivity_type = connectivity_type

        # Connectivity prior scaling hyperparameter
        self.fc_prior_scale = fc_prior_scale

        # Cross-attention 적용 여부 미리 계산
        self._do_cross_map = [(cross_attention_freq > 0) and (i % cross_attention_freq == 0)
                              for i in range(num_layers)]

        # Learnable query tokens [1, 100, 768]
        self.query_tokens = nn.Parameter(torch.randn(1, num_query_tokens, hidden_size))
        nn.init.normal_(self.query_tokens, std=0.02)

        # Connectome-Q-Former blocks
        self.blocks = nn.ModuleList([
            ConnectomeQFormerBlock(hidden_size, num_heads, hidden_size * 4, dropout)
            for _ in range(num_layers)
        ])

        # Final LayerNorm
        self.final_ln = nn.LayerNorm(hidden_size)

        # FC prior 초기화
        self.fc_priors = {}
        if is_fc and subjects is not None and fc_base_dir is not None:
            self._load_fc_priors(subjects, fc_base_dir, num_query_tokens, roi_suffix)

        # Initialize from CLIP weights
        self._init_from_clip(clip_model_name)

    def _init_from_clip(self, clip_model_name):
        """CLIP ViT-B/16에서 weights 가져와서 초기화"""
        print(f"Initializing Q-Former from {clip_model_name}...")

        clip_model = CLIPVisionModel.from_pretrained(clip_model_name)
        clip_layers = clip_model.vision_model.encoder.layers

        for i, (block, clip_layer) in enumerate(zip(self.blocks, clip_layers)):
            # Self-Attention weights
            block.self_ln.weight.data.copy_(clip_layer.layer_norm1.weight.data)
            block.self_ln.bias.data.copy_(clip_layer.layer_norm1.bias.data)

            q_weight = clip_layer.self_attn.q_proj.weight.data
            k_weight = clip_layer.self_attn.k_proj.weight.data
            v_weight = clip_layer.self_attn.v_proj.weight.data
            block.self_attn.in_proj_weight.data.copy_(torch.cat([q_weight, k_weight, v_weight], dim=0))

            q_bias = clip_layer.self_attn.q_proj.bias.data
            k_bias = clip_layer.self_attn.k_proj.bias.data
            v_bias = clip_layer.self_attn.v_proj.bias.data
            block.self_attn.in_proj_bias.data.copy_(torch.cat([q_bias, k_bias, v_bias], dim=0))

            block.self_attn.out_proj.weight.data.copy_(clip_layer.self_attn.out_proj.weight.data)
            block.self_attn.out_proj.bias.data.copy_(clip_layer.self_attn.out_proj.bias.data)

            # FFN weights
            block.ffn_ln.weight.data.copy_(clip_layer.layer_norm2.weight.data)
            block.ffn_ln.bias.data.copy_(clip_layer.layer_norm2.bias.data)
            block.fc1.weight.data.copy_(clip_layer.mlp.fc1.weight.data)
            block.fc1.bias.data.copy_(clip_layer.mlp.fc1.bias.data)
            block.fc2.weight.data.copy_(clip_layer.mlp.fc2.weight.data)
            block.fc2.bias.data.copy_(clip_layer.mlp.fc2.bias.data)

        # Final LayerNorm from CLIP post_layernorm
        self.final_ln.weight.data.copy_(clip_model.vision_model.post_layernorm.weight.data)
        self.final_ln.bias.data.copy_(clip_model.vision_model.post_layernorm.bias.data)

        del clip_model
        print(f"✅ Q-Former initialized from CLIP ViT-B/16 (12 layers, hidden_size=768)")

    def _load_fc_priors(self, subjects, fc_base_dir, num_query_tokens, roi_suffix):
        """FC/SC prior 로드 (FC: arctanh, SC: log1p)"""
        import numpy as np

        conn_tag = self.connectivity_type  # 'FC' or 'SC'
        seq_len = num_query_tokens

        for sub in subjects:
            #fc_path = f"{fc_base_dir}/{sub}/{sub}_{conn_tag}_{roi_suffix}.npy"
            fc_path = f"{fc_base_dir}/{sub}/{sub}_{conn_tag}_{roi_suffix}.npy"

            if not os.path.exists(fc_path):
                print(f"  {conn_tag} prior not found: {fc_path}")
                continue

            fc_prior = np.load(fc_path).astype(np.float32)

            assert fc_prior.shape[0] == fc_prior.shape[1] == seq_len, \
                f"{conn_tag} prior shape mismatch for {sub}: {fc_prior.shape} vs {seq_len}"

            np.fill_diagonal(fc_prior, 0)

            if conn_tag == 'SC':
                fc_prior_z = np.log1p(fc_prior)
            else:
                fc_prior = np.clip(fc_prior, -0.9999, 0.9999)
                fc_prior_z = np.arctanh(fc_prior)

            mask = torch.from_numpy(fc_prior_z)
            mask = mask.unsqueeze(0).repeat(self.num_heads, 1, 1)

            self.fc_priors[sub] = mask

            print(f"  {conn_tag} prior loaded for {sub}: shape={mask.shape}, "
                  f"raw=[{np.load(fc_path).min():.4f}, {np.load(fc_path).max():.4f}] -> "
                  f"transformed=[{fc_prior_z.min():.4f}, {fc_prior_z.max():.4f}]")

        print(f"  Total {len(self.fc_priors)} {conn_tag} priors loaded (pre-expanded for {self.num_heads} heads)")

    def build_mask(self, n_q, n_t, device):
        """Attention mask 생성"""
        L = n_q + n_t
        mask = torch.zeros(L, L, dtype=torch.bool, device=device)
        mask[:n_q, n_q:] = True
        mask[n_q:, :n_q] = True
        return mask

    def _build_fc_prior_mask(self, subject_names, device, dtype=None):
        """Batch의 각 샘플에 대해 해당 subject의 FC prior mask를 생성"""
        masks = []
        for sub_name in subject_names:
            if sub_name in self.fc_priors:
                mask = self.fc_priors[sub_name].to(device=device, dtype=dtype if dtype else self.fc_priors[sub_name].dtype)
            else:
                mask = torch.zeros(self.num_heads, self.num_query_tokens, self.num_query_tokens,
                                 dtype=dtype if dtype else torch.float32, device=device)
            masks.append(mask)

        fc_prior_mask = torch.cat(masks, dim=0)
        fc_prior_mask = fc_prior_mask * self.fc_prior_scale

        return fc_prior_mask


    def forward(self, fmri_emb, clip_hidden=None, use_mask=True, subject_names=None):
        """
        Input:
            fmri_emb [B, 100, 768]
            clip_hidden [B, 257, 768] (optional)
            use_mask: True=Q-T 상호 차단, False=전범위 허용
            subject_names: list of subject names [B]
        Output:
            query_output [B, 100, 768]
        """
        B = fmri_emb.size(0)
        device = fmri_emb.device

        fmri_feats = fmri_emb

        query = self.query_tokens.expand(B, -1, -1)

        if clip_hidden is not None:
            x = torch.cat([query, clip_hidden], dim=1)
            n_q = self.num_query_tokens
            n_t = clip_hidden.size(1)
            attn_mask = self.build_mask(n_q, n_t, device) if use_mask else None
        else:
            x = query
            attn_mask = None

        cross_attn_mask = None
        if self.is_fc and subject_names is not None:
            cross_attn_mask = self._build_fc_prior_mask(subject_names, device, dtype=fmri_feats.dtype)

        for i, block in enumerate(self.blocks):
            x = block(x, fmri_feats,
                     attn_mask=attn_mask,
                     do_cross=self._do_cross_map[i],
                     n_q=self.num_query_tokens,
                     cross_attn_mask=cross_attn_mask,
                     connectivity_type=self.connectivity_type)

        if clip_hidden is not None:
            query_output = x[:, :self.num_query_tokens, :]
        else:
            query_output = x

        query_output = self.final_ln(query_output)

        return query_output


# ============================================================================
# Low-Level Image Decoder
# ============================================================================

class LowLevelDecoder(nn.Module):
    """
    fMRI 임베딩에서 Low-level (blurry) 이미지 생성
    """
    def __init__(self, seq_len=100, embed_dim=768):
        super().__init__()

        self.flatten_dim = seq_len * embed_dim

        self.blin1 = nn.Linear(self.flatten_dim, 64 * 8 * 8, bias=True)
        self.bdropout = nn.Dropout(0.3)
        self.bnorm = nn.GroupNorm(1, 64)

        self.bupsampler = Decoder(
            in_channels=64,
            out_channels=4,
            up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D"],
            block_out_channels=[64, 64, 64],
            layers_per_block=1,
        )

    def forward(self, fmri_emb):
        """
        Input: fmri_emb [B, 100, 768]
        Output: lowlevel_l1 [B, 4, 32, 32]
        """
        B = fmri_emb.size(0)
        x = fmri_emb.view(B, -1)

        lowlevel = self.blin1(x)
        lowlevel = self.bdropout(lowlevel)
        lowlevel = lowlevel.reshape(B, 64, 8, 8).contiguous()
        lowlevel = self.bnorm(lowlevel)

        lowlevel_l1 = self.bupsampler(lowlevel)

        return lowlevel_l1


# ============================================================================
# Output Projection (Linear layer + L2 norm)
# ============================================================================

class OutputProjection(nn.Module):
    """
    Q-Former 출력을 CLIP space로 projection (Transpose 방식)
    """
    def __init__(self, input_tokens=100, output_tokens=257, hidden_size=768):
        super().__init__()
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.hidden_size = hidden_size

        self.proj = nn.Linear(input_tokens, output_tokens)

    def forward(self, x):
        """
        Input: x [B, 100, 768]
        Output: x [B, 257, 768] - L2 normalized
        """
        original_dtype = x.dtype
        x = x.transpose(1, 2)
        x = self.proj(x)
        x = x.transpose(1, 2)
        # FP32로 L2 normalize (FP16 수치 불안정 방지)
        x = F.normalize(x.float(), dim=-1).to(original_dtype)
        return x


# ============================================================================
# Loss Functions
# ============================================================================

class FIRLoss(nn.Module):
    """FIR Loss (fMRI-Image Reconstruction)"""
    def __init__(self):
        super().__init__()
        self.l2 = nn.MSELoss()

    def forward(self, fmri_emb, clip_emb):
        return self.l2(fmri_emb, clip_emb)


class FICLoss(nn.Module):
    """FIC Loss (fMRI-Image Contrastive): BLIP-2 ITC 방식"""
    def __init__(self, temperature=0.07, label_smoothing=0.1):
        super().__init__()
        self.temp = nn.Parameter(torch.ones([]) * temperature)
        self.label_smoothing = label_smoothing

    @torch.no_grad()
    def _gather_features(self, features, accelerator):
        if accelerator is None or accelerator.num_processes == 1:
            return features
        all_features = accelerator.gather(features.contiguous())
        return all_features

    def forward(self, fmri_proj, clip_proj, accelerator=None):
        B = fmri_proj.size(0)
        device = fmri_proj.device

        fmri_norm = fmri_proj
        clip_cls_norm = clip_proj[:, 0, :]

        fmri_norm_all = self._gather_features(fmri_norm, accelerator)
        clip_cls_all = self._gather_features(clip_cls_norm, accelerator)

        if accelerator is not None and accelerator.num_processes > 1:
            rank = accelerator.process_index
            world_size = accelerator.num_processes
        else:
            rank = 0
            world_size = 1

        sim_q2t = torch.einsum('bid,cd->bic', fmri_norm, clip_cls_all)
        sim_fmri2clip, _ = sim_q2t.max(dim=1)

        sim_t2q = torch.einsum('bd,cid->bci', clip_cls_norm, fmri_norm_all)
        sim_clip2fmri, _ = sim_t2q.max(dim=-1)

        # FP32로 temperature scaling + cross_entropy (FP16 overflow 방지)
        sim_fmri2clip_f32 = (sim_fmri2clip / self.temp).float()
        sim_clip2fmri_f32 = (sim_clip2fmri / self.temp).float()

        targets = torch.arange(B, device=device, dtype=torch.long) + rank * B

        loss_fic = (
            F.cross_entropy(sim_fmri2clip_f32, targets, label_smoothing=self.label_smoothing)
            + F.cross_entropy(sim_clip2fmri_f32, targets, label_smoothing=self.label_smoothing)
        ) / 2

        # sim 값은 FP16으로 반환 (FIMLoss에서 사용)
        return loss_fic, sim_fmri2clip, sim_clip2fmri


class FIMLoss(nn.Module):
    """FIM Loss (fMRI-Image Matching): BLIP-2 ITM 방식"""
    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def _gather_features(self, features, accelerator):
        if accelerator is None or accelerator.num_processes == 1:
            return features
        all_features = accelerator.gather(features.contiguous())
        return all_features

    def forward(self, fmri_emb, clip_proj, sim_fmri2clip, sim_clip2fmri,
                qformer, fim_classifier, output_proj, subject_names=None, accelerator=None):
        B = fmri_emb.size(0)
        device = fmri_emb.device

        if accelerator is not None and accelerator.num_processes > 1:
            rank = accelerator.process_index
            world_size = accelerator.num_processes
        else:
            rank = 0
            world_size = 1

        fmri_emb_world = self._gather_features(fmri_emb, accelerator)
        clip_proj_world = self._gather_features(clip_proj, accelerator)

        if subject_names is not None and accelerator is not None and accelerator.num_processes > 1:
            import itertools
            import torch.distributed as dist
            gathered = [None for _ in range(world_size)]
            dist.all_gather_object(gathered, list(subject_names))
            subject_names_world = list(itertools.chain.from_iterable(gathered))
        else:
            subject_names_world = subject_names


        B_world = fmri_emb_world.size(0)
        start_idx = rank * B

        with torch.no_grad():
            sim_fmri2clip_neg = sim_fmri2clip.clone()
            sim_clip2fmri_neg = sim_clip2fmri.clone()

            for b in range(B):
                pos_idx = start_idx + b
                sim_fmri2clip_neg[b, pos_idx] = -10000
                sim_clip2fmri_neg[b, pos_idx] = -10000

            # FP32로 softmax (FP16 overflow 방지)
            weights_fmri2clip = F.softmax(sim_fmri2clip_neg.float(), dim=1)
            weights_clip2fmri = F.softmax(sim_clip2fmri_neg.float(), dim=1)

        clip_neg_indices = []
        for b in range(B):
            neg_idx = torch.multinomial(weights_fmri2clip[b], 1).item()
            clip_neg_indices.append(neg_idx)
        clip_neg_indices = torch.tensor(clip_neg_indices, device=device, dtype=torch.long)
        clip_proj_neg = clip_proj_world[clip_neg_indices]

        fmri_neg_indices = []
        for b in range(B):
            neg_idx = torch.multinomial(weights_clip2fmri[b], 1).item()
            fmri_neg_indices.append(neg_idx)
        fmri_neg_indices = torch.tensor(fmri_neg_indices, device=device, dtype=torch.long)
        fmri_emb_neg = fmri_emb_world[fmri_neg_indices]

        fmri_emb_3b = torch.cat([fmri_emb, fmri_emb, fmri_emb_neg], dim=0)
        clip_proj_3b = torch.cat([clip_proj, clip_proj_neg, clip_proj], dim=0)

        fim_labels = torch.cat([
            torch.ones(B, device=device),
            torch.zeros(2 * B, device=device),
        ], dim=0)

        if subject_names is not None and subject_names_world is not None:
            subject_names_neg = [subject_names_world[i] for i in fmri_neg_indices.cpu().tolist()]
            subject_names_3b = subject_names + subject_names + subject_names_neg
        else:
            subject_names_3b = None

        qformer_out_fim = qformer(fmri_emb_3b, clip_proj_3b, use_mask=False, subject_names=subject_names_3b)
        fmri_proj_fim = output_proj(qformer_out_fim)

        vl_output = fim_classifier(fmri_proj_fim)
        fim_logits = vl_output.mean(dim=1)

        # FP32로 cross_entropy (FP16 overflow 방지)
        loss_fim = F.cross_entropy(fim_logits.float(), fim_labels.long())

        return loss_fim


# ============================================================================
# Complete Model with Bottleneck (BLIP-2 ITM Style)
# ============================================================================

class ConnecToMind2(nn.Module):
    """
    ConnecToMind2 with Bottleneck MLP

    Architecture:
        fMRI [B, 100, input_dim]
            -> (a) Region-level embedding -> [B, 100, 768]
            -> **Bottleneck MLP (768->128->768) -> [B, 100, 768]
            -> (b) Connectome-Q-Former -> [B, 100, 768]
            -> Linear layer -> [B, 257, 768]
            -> L2 norm -> [B, 257, 768]
    """
    def __init__(self, seq_len=100, input_dim=3291, embed_dim=768,
                 num_qformer_layers=12, num_query_tokens=100,
                 is_fc=False, subjects=None, fc_base_dir=None, roi_suffix='schaefer100',
                 fc_prior_scale=1.0, bottleneck_dim=128,
                 use_transformer_bottleneck=False, connectivity_type='FC'):
        super().__init__()

        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.num_query_tokens = num_query_tokens
        self.bottleneck_dim = bottleneck_dim
        self.use_transformer_bottleneck = use_transformer_bottleneck
        self.connectivity_type = connectivity_type
        self.freeze_pre_qformer = False

        # 1. CLIP Image Encoder (ViT-L/14, frozen)
        self.clip_encoder = CLIPImageEncoder(freeze=True)

        # 2. Region-level Embedding
        self.region_embedding = RegionLevelEmbedding(
            seq_len=seq_len,
            input_dim=input_dim,
            embed_dim=embed_dim
        )

        # 3. Bottleneck: MLP vs Transformer Encoder (if문으로 선택)
        if use_transformer_bottleneck:
            self.bottleneck = FCTransformerEncoder(
                d_model=embed_dim,
                nhead=12,
                num_layers=1,
                dropout=0.1,
                is_fc=is_fc,
                subjects=subjects,
                fc_base_dir=fc_base_dir,
                seq_len=seq_len,
                roi_suffix=roi_suffix,
                fc_prior_scale=fc_prior_scale,
                connectivity_type=connectivity_type,
            )
            print(f"  [Bottleneck] FCTransformerEncoder (layers=1, nhead=12)")
        else:
            self.bottleneck = BottleneckMLP(
                embed_dim=embed_dim,
                bottleneck_dim=bottleneck_dim,
                dropout=0.1
            )
            print(f"  [Bottleneck] BottleneckMLP (768 -> {bottleneck_dim} -> 768)")

        # 4. Connectome-Q-Former (initialized from CLIP, with FC prior)
        self.connectome_qformer = ConnectomeQFormer(
            hidden_size=embed_dim,
            num_heads=12,
            num_layers=num_qformer_layers,
            num_query_tokens=num_query_tokens,
            dropout=0.1,
            is_fc=is_fc,
            subjects=subjects,
            fc_base_dir=fc_base_dir,
            roi_suffix=roi_suffix,
            fc_prior_scale=fc_prior_scale,
            connectivity_type=connectivity_type,
        )

        # 5. Output Projection: [B, 100, 768] -> [B, 257, 768]
        self.output_proj = OutputProjection(
            input_tokens=num_query_tokens,
            output_tokens=257,
            hidden_size=embed_dim
        )

        # 6. Low-Level Decoder
        self.low_level_decoder = LowLevelDecoder(seq_len=seq_len, embed_dim=embed_dim)

        # 7. FIM classifier (BLIP-2 ITM Style)
        self.fim_classifier = spectral_norm(nn.Linear(embed_dim, 2))

        # 8. Loss functions
        self.fir_loss_fn = FIRLoss()
        self.fic_loss_fn = FICLoss(temperature=0.07, label_smoothing=0.1)
        self.fim_loss_fn = FIMLoss()

    def forward(self, fmri, images, device, subject_names=None, accelerator=None):
        """
        Training forward

        Input:
            fmri [B, 100, input_dim]
            images [B, 3, 224, 224]
            device: torch device
            subject_names: list of subject names [B]
            accelerator: Accelerator instance

        Output: dict
        """
        B = fmri.size(0)

        # === Image path (CLIP) ===
        clip_proj = self.clip_encoder(images)  # [B, 257, 768]

        # === fMRI path ===
        # (a) Region-level embedding
        fmri_emb = self.region_embedding(fmri)  # [B, 100, 768]

        # (b) Bottleneck (MLP or Transformer Encoder)
        if self.use_transformer_bottleneck:
            fmri_emb = self.bottleneck(fmri_emb, subject_names=subject_names)  # [B, 100, 768]
        else:
            fmri_emb = self.bottleneck(fmri_emb)  # [B, 100, 768]

        # Phase 2: detach로 pre-Q-Former gradient 차단
        if self.freeze_pre_qformer:
            fmri_emb = fmri_emb.detach()

        # === FIR Branch (마스크 O: Q-T 상호 차단) ===
        qformer_out_fir = self.connectome_qformer(fmri_emb, clip_proj, use_mask=True, subject_names=subject_names)
        fmri_proj = self.output_proj(qformer_out_fir)  # [B, 257, 768]

        # === Compute Losses ===

        # 1. FIR Loss
        loss_fir = self.fir_loss_fn(fmri_proj, clip_proj)

        # 2. FIC Loss
        loss_fic, sim_fmri2clip, sim_clip2fmri = self.fic_loss_fn(fmri_proj, clip_proj, accelerator=accelerator)

        # 3. FIM Loss
        loss_fim = self.fim_loss_fn(
            fmri_emb, clip_proj, sim_fmri2clip, sim_clip2fmri,
            qformer=self.connectome_qformer,
            fim_classifier=self.fim_classifier,
            output_proj=self.output_proj,
            subject_names=subject_names,
            accelerator=accelerator
        )

        # 4. Low-level decoder
        lowlevel_l1 = self.low_level_decoder(fmri_emb)

        return {
            "fmri_proj": fmri_proj,
            "clip_proj": clip_proj,
            "lowlevel_l1": lowlevel_l1,
            "loss_fir": loss_fir,
            "loss_fic": loss_fic,
            "loss_fim": loss_fim,
        }

    def inference(self, fmri, subject_names=None):
        """
        Inference (이미지 없이)

        Input:
            fmri [B, 100, input_dim]
            subject_names: list of subject names [B]
        Output:
            fmri_proj: [B, 257, 768]
            lowlevel_l1: [B, 4, 32, 32]
        """
        # (a) Region-level embedding
        fmri_emb = self.region_embedding(fmri)  # [B, 100, 768]

        # (b) Bottleneck (MLP or Transformer Encoder)
        if self.use_transformer_bottleneck:
            fmri_emb = self.bottleneck(fmri_emb, subject_names=subject_names)  # [B, 100, 768]
        else:
            fmri_emb = self.bottleneck(fmri_emb)  # [B, 100, 768]

        # (c) Connectome-Q-Former
        qformer_out = self.connectome_qformer(fmri_emb, subject_names=subject_names)

        # Linear layer + L2 norm
        fmri_proj = self.output_proj(qformer_out)

        # Low-level decoder
        lowlevel_l1 = self.low_level_decoder(fmri_emb)

        return {
            "fmri_proj": fmri_proj,
            "lowlevel_l1": lowlevel_l1,
        }


# ============================================================================
# Model Factory
# ============================================================================

def get_model(args):
    """
    모델 생성 (Bottleneck 버전)

    args 필요 속성:
        - seq_len, input_dim, embed_dim, num_qformer_layers, num_query_tokens
        - cache_dir: pretrained model cache 경로
        - bottleneck_dim (optional): bottleneck dimension (default: 128)
    """
    cache_dir = args.cache_dir

    fc_base_dir = None
    if args.is_fc:
        fc_base_dir = f"{args.root_dir}/{args.fmri_dir}"
        #fc_base_dir = f"{args.root_dir}/{args.fmri_dir}/{args.fmri_detail_dir}"

    # Bottleneck dimension (default: 128)
    bottleneck_dim = getattr(args, 'bottleneck_dim', 128)

    # Transformer bottleneck 옵션
    use_transformer_bottleneck = getattr(args, 'use_transformer_bottleneck', False)

    # Connectivity type (FC or SC)
    connectivity_type = getattr(args, 'connectivity_type', 'SC')

    # connectivity_type에 따라 적절한 scale 선택
    if connectivity_type == 'SC':
        prior_scale = getattr(args, 'sc_prior_scale', 0.5)
    else:
        prior_scale = getattr(args, 'fc_prior_scale', 1.0)

    # 메인 모델
    connectomind2 = ConnecToMind2(
        seq_len=args.seq_len,
        input_dim=args.input_dim,
        embed_dim=args.embed_dim,
        num_qformer_layers=args.num_qformer_layers,
        num_query_tokens=args.num_query_tokens,
        is_fc=args.is_fc,
        subjects=args.subjects,
        fc_base_dir=fc_base_dir,
        roi_suffix=args.roi_suffix,
        fc_prior_scale=prior_scale,
        bottleneck_dim=bottleneck_dim,
        use_transformer_bottleneck=use_transformer_bottleneck,
        connectivity_type=connectivity_type,
    )

    if use_transformer_bottleneck:
        print(f"  ConnecToMind2 with FCTransformerEncoder bottleneck created")
    else:
        print(f"  ConnecToMind2 with BottleneckMLP (768 -> {bottleneck_dim} -> 768) created")

    # High-level reconstruction 용도 -> Versatile Diffusion pipeline
    print("Loading Versatile Diffusion pipeline...")
    try:
        versatile_diffusion = DiffusionPipeline.from_pretrained(
            "shi-labs/versatile-diffusion",
            torch_dtype=torch.float32,
            cache_dir=cache_dir,
            local_files_only=True
        )
        print("✅ Versatile Diffusion loaded (from local cache)")
    except Exception:
        print("  로컬 캐시 없음, 온라인에서 다운로드...")
        try:
            versatile_diffusion = DiffusionPipeline.from_pretrained(
                "shi-labs/versatile-diffusion",
                torch_dtype=torch.float32,
                cache_dir=cache_dir
            )
            print("✅ Versatile Diffusion downloaded and cached")
        except Exception as e:
            print(f"[!] Versatile Diffusion 로딩 실패: {e}")
            versatile_diffusion = None

    # Low-level reconstruction 용도 -> VAE
    print("Loading VAE...")
    try:
        sd_pipe = DiffusionPipeline.from_pretrained(
            "lambdalabs/sd-image-variations-diffusers",
            cache_dir=cache_dir,
            local_files_only=True
        )
        print("✅ VAE loaded (from local cache)")
    except Exception:
        print("  로컬 캐시 없음, 온라인에서 다운로드...")
        sd_pipe = DiffusionPipeline.from_pretrained(
            "lambdalabs/sd-image-variations-diffusers",
            cache_dir=cache_dir
        )
        print("✅ VAE downloaded and cached")
    vae = sd_pipe.vae
    vae.eval().requires_grad_(False)

    # L1 loss
    l1 = nn.L1Loss()

    return {
        "connectomind2": connectomind2,
        "versatile_diffusion": versatile_diffusion,
        "vae": vae,
        "l1": l1,
    }
