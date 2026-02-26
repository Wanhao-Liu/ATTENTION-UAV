from config.baseline import BaselineConfig
from dataclasses import dataclass

@dataclass
class MasacAttnConfig(BaselineConfig):
    exp_name: str = "masac_attn"
    use_attention: bool = True
