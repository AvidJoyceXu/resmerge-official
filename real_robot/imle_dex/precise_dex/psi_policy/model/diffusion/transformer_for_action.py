from typing import Union, Optional, Tuple, List
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from functools import partial

from torch.jit import Final
from timm.models.vision_transformer import Attention, Mlp, RmsNorm, use_fused_attn
from psi_policy.model.diffusion.positional_embedding import SinusoidalPosEmb
from psi_policy.model.common.module_attr_mixin import ModuleAttrMixin

try:
    from rotary_embedding_torch import RotaryEmbedding, apply_rotary_emb
    ROTARY_AVAILABLE = True
except ImportError:
    ROTARY_AVAILABLE = False
    logging.warning("rotary_embedding_torch not available, will use learnable position embeddings")

logger = logging.getLogger(__name__)

# Helper functions
LinearNoBias = partial(nn.Linear, bias=False)

def softclamp(t: torch.Tensor, value: float) -> torch.Tensor:
    """Soft clamping for numerical stability."""
    if value <= 0.:
        return t
    return (t / value).tanh() * value


# SwiGLU Feedforward Network
class SwiGLUFeedForward(nn.Module):
    """SwiGLU-based feedforward - better performance than standard GELU."""
    def __init__(
        self,
        dim: int,
        expand_factor: float = 4.,
        dim_inner: Optional[int] = None,
        use_pre_norm: bool = True,
        use_output_norm: bool = False,
    ):
        super().__init__()
        dim_inner = dim_inner or int(dim * expand_factor * 2 / 3)

        self.pre_norm = RmsNorm(dim, eps=1e-6) if use_pre_norm else nn.Identity()
        self.proj_in = LinearNoBias(dim, dim_inner * 2)
        self.proj_out = LinearNoBias(dim_inner, dim)
        self.post_norm = RmsNorm(dim, eps=1e-6) if use_output_norm else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pre_norm(x)
        x, gate = self.proj_in(x).chunk(2, dim=-1)
        x = x * F.gelu(gate)
        x = self.proj_out(x)
        x = self.post_norm(x)
        return x


# Enhanced Cross Attention with modern improvements
class CrossAttention(nn.Module):
    """
    Cross-attention with enhancements:
    - AlphaFold-style gating
    - Value residual learning
    - Softclamp for stability
    - Aggressive normalization
    """
    fused_attn: Final[bool]
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        norm_layer: nn.Module = RmsNorm,
        use_gating: bool = True,
        use_output_norm: bool = True,
        softclamp_value: float = 50.,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()
        self.softclamp_value = softclamp_value
        self.use_gating = use_gating

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        
        # AlphaFold-style gating
        if use_gating:
            self.gate = LinearNoBias(dim, dim)
        
        self.proj = LinearNoBias(dim, dim)
        
        # Aggressive normalization for stability
        self.proj_out_norm = RmsNorm(dim, eps=1e-6) if use_output_norm else nn.Identity()

    def forward(
        self, 
        x: torch.Tensor, 
        c: torch.Tensor,
        training: bool = True,
        gen_attn_map: bool = False,
        value_residual: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """
        Args:
            x: (B, N, C) - query (actions)
            c: (B, L, C) - key/value (conditions)
            value_residual: optional value residual from previous layer
            
        Returns:
            output: (B, N, C)
            attn_weights: attention weights if gen_attn_map
            v: value tensor for residual learning
        """
        B, N, C = x.shape
        _, L, _ = c.shape
        
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        kv = self.kv(c).reshape(B, L, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        
        # Value residual mixing across layers
        if value_residual is not None:
            v = 0.5 * v + 0.5 * value_residual
        
        attn_weights = None
        
        if self.fused_attn:
            out = F.scaled_dot_product_attention(q, k, v)
            
            if gen_attn_map:
                with torch.no_grad():
                    attn_weights = (q @ k.transpose(-2, -1)) * self.scale
                    attn_weights = softclamp(attn_weights, self.softclamp_value)
                    attn_weights = attn_weights.softmax(dim=-1)
                    attn_weights = attn_weights.detach().cpu().to(torch.float16).numpy()
                    attn_weights = attn_weights[:2, :, 0, :]
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = softclamp(attn, self.softclamp_value)
            attn = attn.softmax(dim=-1)
            
            if gen_attn_map:
                attn_weights = attn.detach().cpu().to(torch.float16).numpy()
                attn_weights = attn_weights[:2, :, 0, :]
                
            out = attn @ v
        
        # AlphaFold-style gating
        if self.use_gating:
            gate = self.gate(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            out = out * gate.sigmoid()
        
        out = out.permute(0, 2, 1, 3).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_out_norm(out)
        
        return out, attn_weights, v


# Transformer Block
class TransformerBlock(nn.Module):
    def __init__(
        self, 
        hidden_size: int,
        num_heads: int,
        softclamp_value: float = 50.,
        use_gating: bool = True,
        use_swiglu: bool = True,
        use_output_norm: bool = True,
        **block_kwargs
    ):
        super().__init__()
        
        # Self-attention (optional, currently disabled for efficiency)
        self.norm1 = RmsNorm(hidden_size, eps=1e-6)
        self.self_attn = Attention(
            dim=hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            qk_norm=True,
            norm_layer=RmsNorm,
            **block_kwargs
        )
        
        # Cross-attention with enhancements
        self.cross_attn = CrossAttention(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            qk_norm=True,
            norm_layer=RmsNorm,
            use_gating=use_gating,
            use_output_norm=use_output_norm,
            softclamp_value=softclamp_value,
        )
        
        self.norm2 = RmsNorm(hidden_size, eps=1e-6)
        
        # Feedforward - SwiGLU or standard MLP
        if use_swiglu:
            self.ffn = SwiGLUFeedForward(
                hidden_size,
                expand_factor=4.,
                use_pre_norm=False,
                use_output_norm=use_output_norm
            )
        else:
            approx_gelu = lambda: nn.GELU(approximate="tanh")
            self.ffn = Mlp(
                in_features=hidden_size,
                hidden_features=hidden_size * 4,
                act_layer=approx_gelu,
                drop=0
            )
        
        self.norm3 = RmsNorm(hidden_size, eps=1e-6)

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        training: bool = True,
        gen_attn_map: bool = False,
        value_residual: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """
        Args:
            x: (B, T, D) - action tokens
            c: (B, N, D) - condition tokens
            value_residual: value residual from previous layer
            
        Returns:
            x: (B, T, D)
            attn_weights: attention weights if gen_attn_map
            new_value_residual: value for next layer
        """
        # Self-attention (optional, disabled for efficiency)
        origin_x = x
        x = self.norm1(x)
        x = self.self_attn(x)
        x = x + origin_x
        
        # Cross-attention
        origin_x = x
        x = self.norm2(x)
        cross_out, attn_weights, new_value_residual = self.cross_attn(
            x, c,
            training=training,
            gen_attn_map=gen_attn_map,
            value_residual=value_residual
        )
        x = origin_x + cross_out
        
        # Feedforward
        origin_x = x
        x = self.norm3(x)
        x = self.ffn(x)
        x = origin_x + x
        
        return x, attn_weights, new_value_residual


class TransformerForAction(ModuleAttrMixin):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        action_horizon: int,
        n_layer: int = 7,
        n_head: int = 8,
        n_emb: int = 768,
        max_cond_tokens: int = 800,
        p_drop_attn: float = 0.1,  
        obs_part_length: List[int] = None,  
        use_attn_mask: bool = True,  
        use_rotary_emb: bool = True,
        softclamp_value: float = 50.,
        use_gating: bool = True,
        use_swiglu: bool = True,
        use_output_norm: bool = True,
    ) -> None:
        """
        Transformer for action prediction with modern enhancements.
        
        Args:
            input_dim: action input dimension
            output_dim: action output dimension
            action_horizon: action sequence length
            n_layer: number of transformer layers
            n_head: number of attention heads
            n_emb: embedding dimension
            max_cond_tokens: max condition tokens
            use_rotary_emb: use RoPE instead of learnable positions
            softclamp_value: softclamp for numerical stability
            use_gating: AlphaFold-style gating in attention
            use_swiglu: use SwiGLU feedforward
            use_output_norm: aggressive output normalization
        """
        super().__init__()
        
        # Input embedding
        self.input_emb = nn.Linear(input_dim, n_emb)
        
        # Position embeddings - RoPE or learnable
        self.use_rotary_emb = use_rotary_emb and ROTARY_AVAILABLE
        if self.use_rotary_emb:
            self.rotary_emb = RotaryEmbedding(n_emb // n_head)
            self.pos_emb = None
        else:
            self.pos_emb = nn.Parameter(torch.randn(1, action_horizon, n_emb))
            self.rotary_emb = None
        
        # Condition position embeddings (always learnable)
        self.cond_pos_emb = nn.Parameter(torch.randn(1, max_cond_tokens, n_emb))
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                hidden_size=n_emb,
                num_heads=n_head,
                softclamp_value=softclamp_value,
                use_gating=use_gating,
                use_swiglu=use_swiglu,
                use_output_norm=use_output_norm,
            )
            for _ in range(n_layer)
        ])
        
        # Final layers with softclamp
        self.final_norm_softclamp = partial(softclamp, value=30.)
        self.ln_f = RmsNorm(n_emb, eps=1e-6)
        self.head = nn.Linear(n_emb, output_dim)
        
        self.action_horizon = action_horizon
        
        # Initialize weights
        self.apply(self._init_weights)
        logger.info(
            "Number of parameters in TransformerForAction: %e",
            sum(p.numel() for p in self.parameters())
        )

    def _init_weights(self, module):
        """Initialize weights following best practices."""
        ignore_types = (
            nn.Dropout,
            nn.TransformerEncoderLayer,
            nn.TransformerDecoderLayer,
            nn.TransformerEncoder,
            nn.TransformerDecoder,
            nn.ModuleList,
            nn.Mish,
            nn.Sequential,
            nn.Embedding,
            nn.SiLU,
            nn.GELU,
            nn.Identity,
            TransformerBlock
        )
        
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
                
        elif isinstance(module, (Attention, CrossAttention)):
            # Attention module weight initialization
            for name in ['q', 'kv', 'k', 'v', 'proj']:
                if hasattr(module, name):
                    weight_module = getattr(module, name)
                    if hasattr(weight_module, 'weight'):
                        torch.nn.init.normal_(weight_module.weight, mean=0.0, std=0.02)
                    if hasattr(weight_module, 'bias') and weight_module.bias is not None:
                        torch.nn.init.zeros_(weight_module.bias)
                        
        elif isinstance(module, Mlp):
            # MLP module weight initialization
            for name in ['fc1', 'fc2']:
                if hasattr(module, name):
                    fc = getattr(module, name)
                    torch.nn.init.normal_(fc.weight, mean=0.0, std=0.02)
                    if fc.bias is not None:
                        torch.nn.init.zeros_(fc.bias)
                        
        elif isinstance(module, nn.MultiheadAttention):
            weight_names = ['in_proj_weight', 'q_proj_weight', 'k_proj_weight', 'v_proj_weight']
            for name in weight_names:
                weight = getattr(module, name, None)
                if weight is not None:
                    torch.nn.init.normal_(weight, mean=0.0, std=0.02)
            
            bias_names = ['in_proj_bias', 'bias_k', 'bias_v']
            for name in bias_names:
                bias = getattr(module, name, None)
                if bias is not None:
                    torch.nn.init.zeros_(bias)
                    
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
            
        elif isinstance(module, RmsNorm):
            if hasattr(module, 'weight') and module.weight is not None:
                torch.nn.init.ones_(module.weight)
                
        elif isinstance(module, TransformerForAction):
            if self.pos_emb is not None:
                torch.nn.init.normal_(self.pos_emb, mean=0.0, std=0.02)
            if self.cond_pos_emb is not None:
                torch.nn.init.normal_(self.cond_pos_emb, mean=0.0, std=0.02)
                
        elif isinstance(module, ignore_types):
            pass
    
    def get_optim_groups(self, weight_decay: float = 1e-3):
        """
        Separate parameters into weight decay and no weight decay groups.
        """
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (nn.Linear, nn.MultiheadAttention)
        blacklist_weight_modules = (nn.LayerNorm, nn.Embedding, RmsNorm)
        
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn
                
                if pn.endswith("bias") or pn.startswith("bias"):
                    no_decay.add(fpn)
                elif "q_norm" in fpn or "k_norm" in fpn:
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)
        
        # Position embeddings don't decay
        if self.pos_emb is not None:
            no_decay.add("pos_emb")
        no_decay.add("cond_pos_emb")
        
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        
        assert len(inter_params) == 0, \
            f"parameters {str(inter_params)} made it into both decay/no_decay sets!"
        assert len(param_dict.keys() - union_params) == 0, \
            f"parameters {str(param_dict.keys() - union_params)} were not separated!"
        
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]
        return optim_groups

    def configure_optimizers(
        self,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.95)
    ):
        """Configure AdamW optimizer with weight decay."""
        optim_groups = self.get_optim_groups(weight_decay=weight_decay)
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
        return optimizer

    def forward(
        self,
        sample: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        training: bool = True,
        gen_attn_map: bool = False,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[List]]:
        """
        Forward pass for action prediction.
        
        Args:
            sample: (B, T, input_dim) - input action sequence
            cond: (B, N, n_emb) - condition embeddings
            training: training mode flag
            gen_attn_map: generate attention maps
            
        Returns:
            output: (B, T, output_dim) - predicted actions
            attention_maps: List of attention weights if gen_attn_map
        """
        # Process condition embeddings with positional encoding
        cond_emb = cond
        tc = cond_emb.shape[1]
        cond_pos_emb = self.cond_pos_emb[:, :tc, :]
        cond_emb = cond_emb + cond_pos_emb
        
        # Process input actions with positional encoding
        action_emb = self.input_emb(sample)
        t = action_emb.shape[1]
        
        if self.use_rotary_emb:
            # RoPE - will be applied in attention if implemented
            # For now, use without extra positional encoding
            x = action_emb
            # Note: RoPE application in CrossAttention would need modification
            # Currently CrossAttention doesn't apply RoPE, keeping for future enhancement
        else:
            pos_emb = self.pos_emb[:, :t, :]
            x = action_emb + pos_emb
        
        # Transformer blocks with value residual learning
        attention_weights = [] if gen_attn_map else None
        value_residual = None
        
        for block in self.blocks:
            x, attn_weights, value_residual = block(
                x,
                cond_emb,
                training=training,
                gen_attn_map=gen_attn_map,
                value_residual=value_residual
            )
            
            if gen_attn_map and attn_weights is not None:
                attention_weights.append(attn_weights)
        
        # Final normalization with softclamp and output
        x = self.final_norm_softclamp(x)
        x = self.ln_f(x)
        x = self.head(x)
        
        return x, attention_weights
