from config.baseline import BaselineConfig
from dataclasses import dataclass

@dataclass
class MasacPerAttnConfig(BaselineConfig):
    exp_name: str = "masac_per_attn"
    use_per: bool = True
    use_attention: bool = True
