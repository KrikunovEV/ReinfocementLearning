import pygame
import sys

pygame.init()
clock = pygame.time.Clock()

screen = pygame.display.set_mode((640, 480))

x = 0

while True:

    msElapsed = clock.tick(60)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    screen.fill((255, 255, 255))

    pygame.draw.rect(screen, (0, 0, 0), (x, 10, 100, 100), 5)
    x += 1

    pygame.display.update()
