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
screen_width = 1920
screen_height = 1080
screen = pygame.display.set_mode((screen_width, screen_height))

# Resource loading function
def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

# Loading assets
asset_background = resource_path('assets\images\\1920x1080p.jpg')
background = pygame.image.load(asset_background)
background = pygame.transform.scale(background, (screen_width, screen_height))

asset_icon = resource_path('assets/images/ufo.png')
icon = pygame.image.load(asset_icon)

asset_sound = resource_path('assets/audios/background_music.mp3')
background_sound = pygame.mixer.music.load(asset_sound)

asset_playerimg = resource_path('assets/images/space-invaders.png')
playerimg = pygame.image.load(asset_playerimg)

asset_bulletimg = resource_path('assets/images/bullet.png')
bulletimg = pygame.image.load(asset_bulletimg)

asset_over_font = resource_path('assets/fonts/RAVIE.TTF')
over_font = pygame.font.Font(asset_over_font, 120)

asset_font = resource_path('assets/fonts/comicbd.ttf')
font = pygame.font.Font(asset_font, 48)

pygame.display.set_caption("Space Invader")
pygame.display.set_icon(icon)

pygame.mixer.music.play(-1)

clock = pygame.time.Clock()

# Player setup
player_width, player_height = playerimg.get_rect().size
playerX = (screen_width - player_width) / 2
playerY = screen_height - player_height - 30

# Enemies setup
enemyimg = []
enemyX = []
enemyY = []
enemyX_change = []
enemyY_change = []
no_of_enemies = 20  # Increased number of enemies

for i in range(no_of_enemies):
    enemy3 = resource_path('assets/images/enemy3.png')
    enemyimg.append(pygame.image.load(enemy3))

    enemy4 = resource_path('assets/images/enemy4.png')
    enemyimg.append(pygame.image.load(enemy4))

    enemyX.append(random.randint(0, screen_width - player_width))
    enemyY.append(random.randint(0, 150))
    enemyX_change.append(5)
    enemyY_change.append(20)  # You can decrease this value to make enemies come near the player faster

# Bullet setup
bullets = []
bulletY_change = 10
bullet_width, bullet_height = bulletimg.get_rect().size

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
def fire_bullet(playerX, playerY):
    if len(bullets) < 20:  # Increased the limit to 10
        bullet = {'x': playerX + player_width / 2 - bullet_width / 2, 'y': playerY - bullet_height}
        bullets.append(bullet)
        screen.blit(bulletimg, (playerX + player_width / 2 - bullet_width / 2, playerY - bullet_height))


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

# Hand detection function
def detect_hand():
    global playerX

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()
    cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    while True:
        success, frame = cap.read()
        if not success:
            continue

        frame = cv2.resize(frame, (1920,1080))

        frame = cv2.flip(frame, 1)

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
                playerX = int(left_hand_x)

                if len(bullets) < 5:
                    fire_bullet(playerX - 20, playerY)
                    fire_bullet(playerX + 20, playerY)

            if right_hand_x is not None:
                playerX = int(right_hand_x)

                if len(bullets) < 5:
                    fire_bullet(playerX - 20, playerY)
                    fire_bullet(playerX + 20, playerY)

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

        for bullet in bullets[:]:
            bullet['y'] -= bulletY_change
            screen.blit(bulletimg, (bullet['x'], bullet['y']))
            if bullet['y'] < 0:
                bullets.remove(bullet)

        for i in range(no_of_enemies):
            if enemyY[i] > playerY:
                game_over_text()
                in_game = False
                break

            enemyX[i] += enemyX_change[i]
            if enemyX[i] <= 0:
                enemyX_change[i] = 5
                enemyY[i] += enemyY_change[i]
            elif enemyX[i] >= screen_width - player_width:
                enemyX_change[i] = -5
                enemyY[i] += enemyY_change[i]

            for bullet in bullets:
                collision = isCollision(enemyX[i], enemyY[i], bullet['x'], bullet['y'])
                if collision:
                    bullets.remove(bullet)
                    score += 1
                    enemyX[i] = random.randint(0, screen_width - player_width)
                    enemyY[i] = random.randint(0, 150)
                    break

            enemy(enemyX[i], enemyY[i], i)

        player(playerX, playerY)
        show_score()

        pygame.display.update()
        clock.tick(120)

hand_thread = threading.Thread(target=detect_hand)
hand_thread.start()

gameloop()
