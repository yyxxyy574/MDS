import pygame

class Cell(pygame.Rect):
    def __init__(self, x, y, src_x, src_y, width, height):
        super().__init__(x, y, width, height)
        self.src_rect = pygame.Rect(
            src_x * width,
            src_y * height,
            width,
            height,
        )