# -*- coding: utf-8 -*-
from config.baseline import BaselineConfig
from dataclasses import dataclass

@dataclass
class MasacPerConfig(BaselineConfig):
    exp_name: str = "masac_per"
    use_per: bool = True
