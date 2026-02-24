# -*- coding: utf-8 -*-
#开发者：Bright Fang
#开发时间：2022/9/17 18:22
import pygame
from env.assignment import constants as C
from env.assignment import tools
import os

pygame.init()
try:
    pygame.mixer.init()
except pygame.error:
    pass
SCREEN=pygame.display.set_mode((C.SCREEN_W, C.SCREEN_H))

pygame.display.set_caption("eee")

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_IMAGE_PATH = os.path.join(_PROJECT_ROOT, 'env', 'source', 'image')
_MUSIC_PATH = os.path.join(_PROJECT_ROOT, 'env', 'source', 'music')

GRAPHICS=tools.load_graphics(_IMAGE_PATH)
SOUND=tools.load_sound(_MUSIC_PATH)
