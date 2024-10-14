import pygame
from pygame.locals import *

pygame.init()
pygame.mixer.init()
pygame.font.init()

FPS = 3
WINDOWWIDTH = 640
WINDOWHEIGHT = 480
CELLSIZE = 20
assert WINDOWHEIGHT % CELLSIZE == 0, "Window Height must be a multiple of Cell Size"
assert WINDOWWIDTH % CELLSIZE == 0, "Window Width must be a multiple of Cell Size"
CELLWIDTH = int(WINDOWWIDTH / CELLSIZE)
CELLHEIGHT = int(WINDOWHEIGHT / CELLSIZE)

#Colour Codes
#			 R    G    B
WHITE    = (255, 255, 255)
BLACK    = (0,     0,   0)
RED      = (255,   0,   0)
GREEN    = (0,   255,   0)
DARKGREEN= (0,   155,   0)
DARKGRAY = (40,   40,  40)
YELLOW   = (255, 255,   0)

BGCOLOR = BLACK

#Control Keys
UP = 'up'
DOWN = 'down'
LEFT = 'left'
RIGHT = 'right'

HEAD = 0 #Index of the snake's head

#Game Sounds
APPLEEATSOUND = pygame.mixer.Sound(r"sounds/appleEatSound.wav")
BGMUSIC = pygame.mixer.music.load(r"sounds/bgmusic.mid")

def levelSelect():
	global FPS
	if level == "EASY":
		FPS = 4
	elif level == "MEDIUM":
		FPS = 7
	elif level == "HARD":
		FPS = 10
