import random
import pygame
import sys
from pygame.locals import *
from settingsSnakeFun import *

def main():
	global CLOCK, SCREEN, FONT

	pygame.init()
	CLOCK = pygame.time.Clock()
	SCREEN = pygame.display.set_mode((WINDOWWIDTH, WINDOWHEIGHT))
	FONT = pygame.font.Font('freesansbold.ttf', 18)
	pygame.display.set_caption('Snake Game')

	showStartScreen()
	while True:
		pygame.mixer.music.play(-1,0.0)
		runGame()
		pygame.mixer.music.stop()
		showGameOverScreen()

def runGame():
	#Set a random starting point
	startx = random.randint(5, CELLWIDTH - 6)
	starty = random.randint(5, CELLHEIGHT - 6)
	global wormCoords
	wormCoords = [{'x' : startx, 'y' : starty}, {'x': startx - 1, 'y':starty}, {'x':startx - 2, 'y':starty}]
	direction = RIGHT

	apple = getRandomLocation()

	while True:
		for event in pygame.event.get():
			if event.type == QUIT:
				terminate()
			elif event.type == KEYDOWN:
				if (event.key == K_LEFT or event.key == K_a) and direction != RIGHT:
					direction = LEFT
				elif (event.key == K_RIGHT or event.key == K_d) and direction != LEFT:
					direction = RIGHT
				elif (event.key == K_UP or event.key == K_w) and direction != DOWN:
					direction = UP
				elif (event.key == K_DOWN or event.key == K_s) and direction != UP:
					direction = DOWN
				elif event.key == K_ESCAPE:
					terminate()
#Collision Detection
		#Check Collision with edges
		if wormCoords[HEAD]['x'] == -1 or wormCoords[HEAD]['x'] == CELLWIDTH or wormCoords[HEAD]['y'] == -1 or wormCoords[HEAD]['y'] == CELLHEIGHT:
			return
		#Check Collision with snake's body
		for wormBody in wormCoords[1:]:
			if wormBody['x'] == wormCoords[HEAD]['x'] and wormBody['y'] == wormCoords[HEAD]['y']:
				return
		#Check Collision with Apple
		if wormCoords[HEAD]['x'] == apple['x'] and wormCoords[HEAD]['y'] == apple['y']:
			APPLEEATSOUND.play()
			apple = getRandomLocation()
		else:
			del wormCoords[-1]

#Moving the Snake
		if direction == UP:
			newHead = {'x': wormCoords[HEAD]['x'], 'y': wormCoords[HEAD]['y'] - 1}
		elif direction == DOWN:
			newHead = {'x': wormCoords[HEAD]['x'], 'y': wormCoords[HEAD]['y'] + 1}
		elif direction == RIGHT:
			newHead = {'x': wormCoords[HEAD]['x'] + 1, 'y': wormCoords[HEAD]['y']}
		elif direction == LEFT:
			newHead = {'x': wormCoords[HEAD]['x'] - 1, 'y': wormCoords[HEAD]['y']}
		wormCoords.insert(0, newHead)

#Drawing the Screen
		SCREEN.fill(BGCOLOR)
		drawGrid()
		drawWorm(wormCoords)
		drawApple(apple)
		drawScore((len(wormCoords) - 3) * 10)
		pygame.display.update()
		CLOCK.tick(FPS)

def getTotalScore():
	return ((len(wormCoords) - 3) * 10)

def drawPressKeyMsg():
	pressKeyText = FONT.render('Press A Key To Play', True, YELLOW)
	pressKeyRect = pressKeyText.get_rect()
	pressKeyRect.center = (WINDOWWIDTH - 200, WINDOWHEIGHT - 100)
	SCREEN.blit(pressKeyText, pressKeyRect)

def drawSettingsMsg():
	SCREEN.blit(SETTINGSBUTTON, (WINDOWWIDTH - SETTINGSBUTTON.get_width(), WINDOWHEIGHT - SETTINGSBUTTON.get_height()))

def checkForKeyPress():
	if len(pygame.event.get(QUIT)) > 0:
		terminate()

	keyUpEvents = pygame.event.get(KEYUP)
	if len(keyUpEvents) == 0:
		return None
	if keyUpEvents[0].key == K_ESCAPE:
		terminate()
	return keyUpEvents[0].key

def showStartScreen():
	titlefont = pygame.font.Font('freesansbold.ttf', 100)
	titleText = titlefont.render('SNAKE FUN', True, DARKGREEN)
	while True:
		SCREEN.fill(BGCOLOR)
		titleTextRect = titleText.get_rect()
		titleTextRect.center = (WINDOWWIDTH / 2, WINDOWHEIGHT / 2)
		SCREEN.blit(titleText, titleTextRect)

		drawPressKeyMsg()
		if checkForKeyPress():
			pygame.event.get()
			return
		pygame.display.update()
		CLOCK.tick(FPS)

def terminate():
	pygame.quit()
	sys.exit()

def getRandomLocation():
	return {'x': random.randint(0, CELLWIDTH - 1), 'y': random.randint(0, CELLHEIGHT - 1)}

def showGameOverScreen():
	gameOverFont = pygame.font.Font('freesansbold.ttf', 100)
	gameOverText = gameOverFont.render('Game Over', True, WHITE)
	gameOverRect = gameOverText.get_rect()
	totalscoreFont = pygame.font.Font('freesansbold.ttf', 40)
	totalscoreText = totalscoreFont.render('Total Score: %s' % (getTotalScore()), True, WHITE)
	totalscoreRect = totalscoreText.get_rect()
	totalscoreRect.midtop = (WINDOWWIDTH/2, 150)
	gameOverRect.midtop = (WINDOWWIDTH/2, 30)
	SCREEN.fill(BGCOLOR)
	SCREEN.blit(gameOverText, gameOverRect)
	SCREEN.blit(totalscoreText, totalscoreRect)
	drawPressKeyMsg()
	pygame.display.update()
	pygame.time.wait(1000)
	checkForKeyPress()

	while True:
		if checkForKeyPress():
			pygame.event.get()
			return

def drawScore(score):
	scoreText = FONT.render('Score: %s' % (score), True, WHITE)
	scoreRect = scoreText.get_rect()
	scoreRect.center = (WINDOWWIDTH - 100, 30)
	SCREEN.blit(scoreText, scoreRect)

def drawWorm(wormCoords):
	x = wormCoords[HEAD]['x'] * CELLSIZE
	y = wormCoords[HEAD]['y'] * CELLSIZE
	wormHeadRect = pygame.Rect(x, y, CELLSIZE, CELLSIZE)
	pygame.draw.rect(SCREEN, YELLOW, wormHeadRect)

	for coord in wormCoords[1:]:
		x = coord['x'] * CELLSIZE
		y = coord['y'] * CELLSIZE
		wormSegmentRect = pygame.Rect(x, y, CELLSIZE, CELLSIZE)
		pygame.draw.rect(SCREEN, GREEN, wormSegmentRect)

def drawApple(coord):
	x = coord['x'] * CELLSIZE
	y = coord['y'] * CELLSIZE
	appleRect = pygame.Rect(x, y, CELLSIZE, CELLSIZE)
	pygame.draw.rect(SCREEN, RED, appleRect)

def drawGrid():
	for x in range(0, WINDOWWIDTH, CELLSIZE):
		pygame.draw.line(SCREEN, DARKGRAY, (x, 0), (x, WINDOWHEIGHT))
	for y in range(0, WINDOWHEIGHT, CELLSIZE):
		pygame.draw.line(SCREEN, DARKGRAY, (0, y), (WINDOWWIDTH, y))

if __name__ == '__main__':
	main()
