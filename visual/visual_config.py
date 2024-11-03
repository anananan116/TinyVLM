from dataclasses import dataclass
from typing import Optional

@dataclass
class EVAVisionConfig:
    """Configuration class for EVA Vision Transformer model.
    
    All the default values are set according to the EVA-CLIP-E-14-plus configuration.
    """
    
    image_size: int = 448
    patch_size: int = 14
    width: int = 1792
    layers: int = 64
    head_width: int = 112
    mlp_ratio: float = 8.571428571428571
    qkv_bias: bool = True
    drop_path_rate: float = 0.0
    xattn: bool = False
    postnorm: bool = True
    
    # Additional configuration parameters not directly used in model initialization
    eva_model_name: str = "eva-clip-E-14-plus"
    intermediate_size: int = 15360
    layer_norm_eps: float = 1e-6
    n_query: int = 256
    v_query: int = 64