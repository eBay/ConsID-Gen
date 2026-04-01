import torch
import torch.nn as nn

from .wan_video_dit import AttentionModule, RMSNorm


class ProjectionMLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x):
        return self.proj(x)


class DualStreamFusionAttention(nn.Module):
    def __init__(
        self,
        dim_a: int,
        dim_b: int,
        num_attention_heads: int = 16,
        attention_head_dim: int = 128,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.num_heads = num_attention_heads
        self.head_dim = attention_head_dim

        self.q = nn.Linear(dim_a, dim_a)
        self.k = nn.Linear(dim_a, dim_a)
        self.v = nn.Linear(dim_a, dim_a)
        self.norm_q = RMSNorm(dim_a, eps=eps)
        self.norm_k = RMSNorm(dim_a, eps=eps)

        self.add_q = nn.Linear(dim_b, dim_b)
        self.add_k = nn.Linear(dim_b, dim_b)
        self.add_v = nn.Linear(dim_b, dim_b)
        self.norm_add_q = RMSNorm(dim_b, eps=eps)
        self.norm_add_k = RMSNorm(dim_b, eps=eps)

        self.attn = AttentionModule(self.num_heads)

        self.o = nn.Linear(dim_a, dim_a)
        self.add_o = nn.Linear(dim_b, dim_b)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        x_q, x_k, x_v = self.norm_q(self.q(x)), self.norm_k(self.k(x)), self.v(x)
        y_q, y_k, y_v = self.norm_add_q(self.add_q(y)), self.norm_add_k(self.add_k(y)), self.add_v(y)
        seq_x_clip = x_q.shape[1]

        joint_q = torch.cat([x_q, y_q], dim=1)
        joint_k = torch.cat([x_k, y_k], dim=1)
        joint_v = torch.cat([x_v, y_v], dim=1)

        joint_attn_out = self.attn(joint_q, joint_k, joint_v)

        x_attn_output = self.o(joint_attn_out[:, :seq_x_clip, :])
        y_attn_output = self.add_o(joint_attn_out[:, seq_x_clip:, :])

        return x_attn_output, y_attn_output


class DualStreamFusionBlock(nn.Module):
    def __init__(
        self,
        dim: int = 2048,
        num_attention_heads: int = 16,
        attention_head_dim: int = 128,
        eps: float = 1e-6,
    ):
        super().__init__()
        assert attention_head_dim * num_attention_heads == dim, "model_dim must equal dim_b after clip_proj."

        self.x_in_norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.y_in_norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.attn = DualStreamFusionAttention(dim, dim, num_attention_heads, attention_head_dim, eps)

        self.x_gate = nn.Parameter(torch.tensor(1.0))
        self.y_gate = nn.Parameter(torch.tensor(1.0))

        self.x_ffn = nn.Sequential(
            nn.LayerNorm(dim, eps=eps),
            nn.Linear(dim, dim * 2),
            nn.GELU(approximate="tanh"),
            nn.Linear(dim * 2, dim),
        )
        self.y_ffn = nn.Sequential(
            nn.LayerNorm(dim, eps=eps),
            nn.Linear(dim, dim * 2),
            nn.GELU(approximate="tanh"),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        x_norm = self.x_in_norm(x)
        y_norm = self.y_in_norm(y)
        x_att_output, y_att_output = self.attn(x_norm, y_norm)

        x_out = x + self.x_gate * x_att_output
        y_out = y + self.y_gate * y_att_output

        x_out = x_out + self.x_ffn(x_out)
        y_out = y_out + self.y_ffn(y_out)
        return x_out, y_out


class GeometrySelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)
        self.attn = AttentionModule(num_heads)

    def forward(self, x):
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(x))
        v = self.v(x)
        x = self.attn(q, k, v)
        return self.o(x)


class GeometryCrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)
        self.o = nn.Linear(dim, dim)
        self.attn = AttentionModule(num_heads)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(y))
        v = self.v(y)
        x = self.attn(q, k, v)
        return self.o(x)


class ViewAssistedGeometryRefiner(nn.Module):
    def __init__(self, dim: int = 2048, num_heads: int = 16, eps: float = 1e-6):
        super().__init__()
        self.self_attn = GeometrySelfAttention(dim, num_heads, eps)
        self.cross_attn1 = GeometryCrossAttention(dim, num_heads, eps)
        self.cross_attn2 = GeometryCrossAttention(dim, num_heads, eps)
        self.cross_attn3 = GeometryCrossAttention(dim, num_heads, eps)
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim, eps=eps),
            nn.Linear(dim, dim * 4),
            nn.GELU(approximate="tanh"),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, clip_feat, vggt_feat, vggt_ref_feats):
        mem = vggt_feat + self.self_attn(vggt_feat)
        vggt_ref_feat1, vggt_ref_feat2 = vggt_ref_feats[:, 0], vggt_ref_feats[:, 1]
        mem = self.cross_attn1(mem, vggt_ref_feat1)
        mem = self.cross_attn2(mem, vggt_ref_feat2)
        mem_out = mem + self.ffn(mem)
        clip_feat_out = clip_feat + self.cross_attn3(clip_feat, mem_out)
        return clip_feat_out, mem_out


class MultiModalTextVisualModule(nn.Module):
    def __init__(
        self,
        dim: int = 2048,
        num_attention_heads: int = 16,
        attention_head_dim: int = 128,
        eps: float = 1e-6,
    ):
        super().__init__()

        self.x_in_norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.y_in_mlp = ProjectionMLP(4096, 2048)
        self.y_in_norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)

        self.attn = DualStreamFusionAttention(dim, dim, num_attention_heads, attention_head_dim, eps)

        self.x_gate = nn.Parameter(torch.tensor(1.0))
        self.y_gate = nn.Parameter(torch.tensor(1.0))

        self.x_ffn = nn.Sequential(
            nn.LayerNorm(dim, eps=eps),
            nn.Linear(dim, dim * 2),
            nn.GELU(approximate="tanh"),
            nn.Linear(dim * 2, dim),
        )
        self.y_ffn = nn.Sequential(
            nn.LayerNorm(dim, eps=eps),
            nn.Linear(dim, dim * 2),
            nn.GELU(approximate="tanh"),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        x_norm = self.x_in_norm(x)
        y_norm = self.y_in_norm(self.y_in_mlp(y))
        x_att_output, y_att_output = self.attn(x_norm, y_norm)

        x_out = x_norm + self.x_gate * x_att_output
        y_out = y_norm + self.y_gate * y_att_output

        x_out = x_out + self.x_ffn(x_out)
        y_out = y_out + self.y_ffn(y_out)
        return x_out, y_out


class ConsIDGenAdapter(nn.Module):
    _checkpoint_prefix_aliases = ("pipe.hybridadaptionformer.",)
    _checkpoint_key_aliases = {
        "dual_visual_encoder.": "visual_geometric_blocks.",
        "refmixer.": "view_assisted_refiner.",
        "text_visual_mm_adapter.": "text_visual_module.",
    }

    def __init__(
        self,
        dim_a: int = 1280,
        dim_b: int = 2048,
        dim_c: int = 4096,
        num_attention_heads: int = 16,
        attention_head_dim: int = 128,
        num_dv_layers: int = 1,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.clip_in = ProjectionMLP(in_dim=dim_a, out_dim=dim_b)
        self.visual_geometric_blocks = nn.ModuleList(
            [DualStreamFusionBlock(dim_b, num_attention_heads, attention_head_dim, eps) for _ in range(num_dv_layers)]
        )
        self.view_assisted_refiner = ViewAssistedGeometryRefiner(dim_b, num_attention_heads, eps)
        self.text_visual_module = MultiModalTextVisualModule(dim_b, num_attention_heads, eps)
        self.visual_out_mlp = ProjectionMLP(dim_b, dim_a)
        self.text_out_mlp = ProjectionMLP(dim_b, dim_c)

    def forward(self, vggt_feats: torch.Tensor, text_feats: torch.Tensor, clip_feat: torch.Tensor):
        clip_in_feat = self.clip_in(clip_feat)
        if vggt_feats.dim() == 3:
            vggt_feats = vggt_feats.unsqueeze(1)
        vggt_in_feat, vggt_ref_feats = vggt_feats[:, 0], vggt_feats[:, 1:]

        for blk in self.visual_geometric_blocks:
            clip_in_feat, vggt_in_feat = blk(clip_in_feat, vggt_in_feat)

        visual_feat, mem_feat = self.view_assisted_refiner(clip_in_feat, vggt_in_feat, vggt_ref_feats)
        visual_feat_out, text_feat_out = self.text_visual_module(visual_feat, text_feats)

        visual_feat_out = clip_feat + self.visual_out_mlp(visual_feat_out)
        text_feat_out = text_feats + self.text_out_mlp(text_feat_out)

        return {
            "visual_conditioned_on_text": visual_feat_out,
            "text_conditioned_on_visual": text_feat_out,
        }


# Backward-compatible aliases: keep external imports and old call sites stable.
MLP = ProjectionMLP
DualStreamJointAttention = DualStreamFusionAttention
DualStreamJointBlock = DualStreamFusionBlock
RefMixerSelfAtt = GeometrySelfAttention
RefMixerCrossAtt = GeometryCrossAttention
ReferenceSelfAttention = GeometrySelfAttention
ReferenceCrossAttention = GeometryCrossAttention
RefMixer = ViewAssistedGeometryRefiner
ReferenceFeatureMixer = ViewAssistedGeometryRefiner
TextVisualMMAdapter = MultiModalTextVisualModule
TextVisualFusionAdapter = MultiModalTextVisualModule
HybridAdaptionFormer = ConsIDGenAdapter
HybridAdaptationFormer = ConsIDGenAdapter
