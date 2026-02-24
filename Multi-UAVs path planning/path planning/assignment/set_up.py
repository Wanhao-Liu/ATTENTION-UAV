# -*- coding: utf-8 -*-
#开发者：Bright Fang
#开发时间：2022/9/17 18:22
import pygame
from assignment import  constants as C
from assignment import tools
pygame.init()
try:
    pygame.mixer.init()
except pygame.error:
    pass
SCREEN=pygame.display.set_mode((C.SCREEN_W, C.SCREEN_H))

pygame.display.set_caption("eee")

GRAPHICS=tools.load_graphics('/home/lwh/UAV-path-planning/Multi-UAVs path planning/path planning/assignment/source/image')

SOUND=tools.load_sound('/home/lwh/UAV-path-planning/Multi-UAVs path planning/path planning/assignment/source/music')