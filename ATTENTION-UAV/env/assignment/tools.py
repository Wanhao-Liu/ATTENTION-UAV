# -*- coding: utf-8 -*-
#开发者：Bright Fang
#开发时间：2022/9/17 18:23
import pygame
from env.assignment import constants as C
import os

def load_graphics(path,accept=('.jpg','.png','.bmp','.gif')):
    graphics={}
    for pic in os.listdir(path):
        name,ext=os.path.splitext(pic)
        if ext.lower() in accept:
            img=pygame.image.load(os.path.join(path,pic))
            if img.get_alpha():
                img=img.convert_alpha()
            else:
                img=img.convert()
            graphics[name]=img
    return graphics

def load_sound(path,accept=('.wav','.mp3')):
    sound={}
    if not pygame.mixer.get_init():
        return sound
    for pic in os.listdir(path):
        name,ext=os.path.splitext(pic)
        if ext.lower() in accept:
            sou=pygame.mixer.Sound(os.path.join(path,pic))
            sound[name]=sou
    return sound
