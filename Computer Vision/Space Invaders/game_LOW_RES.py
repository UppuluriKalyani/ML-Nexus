import pygame
import random
import math
import sys
import os
import cv2
import mediapipe as mp
import threading

pygame.init()

# Screen setup
screen_width = 800
screen_height = 600
screen = pygame.display.set_mode((screen_width, screen_height))

# Resource loading function
def resource_path(relative_path):
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

# Loading assets
asset_background = resource_path('assets\images\800x600p.jpg')
background = pygame.image.load(asset_background)

asset_icon = resource_path('assets/images/ufo.png')
icon = pygame.image.load(asset_icon)

asset_sound = resource_path('assets/audios/background_music.mp3')
background_sound = pygame.mixer.music.load(asset_sound)

asset_playerimg = resource_path('assets/images/space-invaders.png')
playerimg = pygame.image.load(asset_playerimg)

asset_bulletimg = resource_path('assets/images/bullet.png')
bulletimg = pygame.image.load(asset_bulletimg)

asset_over_font = resource_path('assets/fonts/RAVIE.TTF')
over_font = pygame.font.Font(asset_over_font, 60)

asset_font = resource_path('assets/fonts/comicbd.ttf')
font = pygame.font.Font(asset_font, 32)

# Setting up the window
pygame.display.set_caption("Space Invader")
pygame.display.set_icon(icon)

# Background music
pygame.mixer.music.play(-1)

# Clock for controlling frame rate
clock = pygame.time.Clock()

# Player setup
playerX = 370
playerY = 470

# Enemies setup
enemyimg = []
enemyX = []
enemyY = []
enemyX_change = []
enemyY_change = []
no_of_enemies = 10

for i in range(no_of_enemies):
    enemy3 = resource_path('assets/images/enemy3.png')
    enemyimg.append(pygame.image.load(enemy3))

    enemy4 = resource_path('assets/images/enemy4.png')
    enemyimg.append(pygame.image.load(enemy4))

    enemyX.append(random.randint(0, 736))
    enemyY.append(random.randint(0, 150))
    enemyX_change.append(5)
    enemyY_change.append(20)

# Bullet setup
bullets = []  # List to hold multiple bullets
bulletY_change = 10

# Score setup
score = 0

# Function to display score
def show_score():
    score_value = font.render("SCORE " + str(score), True, (255, 255, 255))
    screen.blit(score_value, (10, 10))

# Function to draw player
def player(x, y):
    screen.blit(playerimg, (x, y))

# Function to draw enemy
def enemy(x, y, i):
    screen.blit(enemyimg[i], (x, y))

# Function to fire bullet
def fire_bullet(x, y):
    bullet = {'x': x + 16, 'y': y + 10}  # Dictionary to hold bullet position
    bullets.append(bullet)  # Add bullet to the list
    screen.blit(bulletimg, (x + 16, y + 10))

# Function to check collision between bullet and enemy
def isCollision(enemyX, enemyY, bulletX, bulletY):
    distance = math.sqrt((math.pow(enemyX - bulletX, 2)) + (math.pow(enemyY - bulletY, 2)))
    if distance < 27:
        return True
    else:
        return False

# Function to display game over text
def game_over_text():
    over_text = over_font.render("GAME OVER", True, (255, 255, 255))
    text_rect = over_text.get_rect(center=(int(screen_width / 2), int(screen_height / 2)))
    screen.blit(over_text, text_rect)


def detect_hand():
    global playerX

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()
    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
            continue

        frame = cv2.flip(frame, 1)  # Flip the frame horizontally

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        left_hand_x = None
        right_hand_x = None

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for idx, landmark in enumerate(hand_landmarks.landmark):
                    if idx == mp_hands.HandLandmark.INDEX_FINGER_TIP:
                        if landmark.x * frame.shape[1] < frame.shape[1] / 2:
                            left_hand_x = landmark.x * frame.shape[1]
                        else:
                            right_hand_x = landmark.x * frame.shape[1]

            if left_hand_x is not None:
                if left_hand_x < playerX:
                    playerX -= 15
                elif left_hand_x > playerX:
                    playerX += 15

                if len(bullets) < 5:  # Limit bullets to 5
                    fire_bullet(playerX - 20, 454)  # Fire bullet line 1
                    fire_bullet(playerX + 20, 454)  # Fire bullet line 2

            if right_hand_x is not None:
                if right_hand_x < playerX:
                    playerX -= 15
                elif right_hand_x > playerX:
                    playerX += 15

                if len(bullets) < 5:  # Limit bullets to 5
                    fire_bullet(playerX - 20, 454)  # Fire bullet line 1
                    fire_bullet(playerX + 20, 454)  # Fire bullet line 2

        cv2.imshow("Hand Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# Game loop
def gameloop():
    global score

    in_game = True
    while in_game:
        screen.fill((0, 0, 0))
        screen.blit(background, (0, 0))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                in_game = False
                pygame.quit()
                sys.exit()

        for bullet in bullets[:]:  # Loop through a copy of the bullets list
            bullet['y'] -= bulletY_change
            screen.blit(bulletimg, (bullet['x'], bullet['y']))
            if bullet['y'] < 0:
                bullets.remove(bullet)  # Remove bullets when they go off the screen

        for i in range(no_of_enemies):
            if enemyY[i] > 440:
                for j in range(no_of_enemies):
                    enemyY[j] = 2000
                game_over_text()

            enemyX[i] += enemyX_change[i]
            if enemyX[i] <= 0:
                enemyX_change[i] = 5
                enemyY[i] += enemyY_change[i]
            elif enemyX[i] >= 736:
                enemyX_change[i] = -5
                enemyY[i] += enemyY_change[i]

            for bullet in bullets:  # Check for collision with each bullet
                collision = isCollision(enemyX[i], enemyY[i], bullet['x'], bullet['y'])
                if collision:
                    bullets.remove(bullet)
                    score += 1
                    enemyX[i] = random.randint(0, 736)
                    enemyY[i] = random.randint(0, 150)
                    break  # No need to check collision for other bullets if one hits an enemy

            enemy(enemyX[i], enemyY[i], i)

        player(playerX, playerY)
        show_score()

        pygame.display.update()
        clock.tick(120)

# Start hand detection in a separate thread
hand_thread = threading.Thread(target=detect_hand)
hand_thread.start()

# Start the game loop
gameloop()
