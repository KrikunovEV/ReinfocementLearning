import pygame
import sys
import numpy as np

from core.interfaces.IEnvironment import IEnvironment
from enum import IntEnum, Enum
from random import randint
from dataclasses import dataclass


class TIenv(IEnvironment):

    @dataclass
    class Observation:
        done: bool
        reward: float
        value: float
        cells_type: np.ndarray
        entity_map: np.ndarray
        treasure_map: np.ndarray
        value_map: np.ndarray

        def __str__(self):
            cell_type_list = ""
            for type in TIenv.Cell.Type:
                cell_type_list += type.name + "=" + str(type.value) + ", "

            return "\ndone: {done}\nreward: {reward}\nvalue: {value}\ncells_type({cell_type_list}):\n{cells_type}\n" \
                   "entity_map(NOBODY=0, AGENT=1, ):\n{entity_mep}\ntreasure_map(scalar):\n{treasure_map}\n" \
                   "value_map(scalar):\n{value_map}\n".format(
                    done=self.done, reward=self.reward, value=self.value, cell_type_list=cell_type_list,
                    cells_type=self.cells_type, entity_mep=self.entity_map, treasure_map=self.treasure_map,
                    value_map=self.value_map)

    class Cell:

        class Type(Enum):
            EMPTY = 0
            TREASURE = 1

        treasure_residue = 0
        type = Type.EMPTY
        value = 0

    class Entity:

        class Actions(IntEnum):
            UP = 0,
            RIGHT = 1,
            DOWN = 2,
            LEFT = 3,
            DIG = 4,
            STAY = 5

        def __init__(self, x=0, y=0, coalition=1):
            self.x = x
            self.y = y
            self.coalition = coalition

    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    RED = (255, 0, 0)

    TREASURE_MAX = 25
    TREASURE_MIN = 5

    def __init__(self, frame_rate=60, num_marks=2):
        self.frame_rate = frame_rate
        self.window_size = (1000, 600)

        self.cell_size = (50, 50)
        self.entity_size = (40, 40)

        self.num_marks = num_marks

        pygame.init()
        pygame.display.set_caption('Treasure Island')
        self.clock = pygame.time.Clock()
        self.screen = pygame.display.set_mode(self.window_size)

    def __del__(self):
        pygame.quit()

    def reset(self):

        self.cells = [[self.Cell() for x in range(int(self.window_size[0] / self.cell_size[0]))]
                      for y in range(int(self.window_size[1] / self.cell_size[1]))]

        self.entity = self.Entity()

        self.max_steps = (self.TREASURE_MAX + self.TREASURE_MIN) / 2  # mean (expected) treasure count in one cell
        self.max_steps *= 9  # such cells is 9 in one treasure place
        self.max_steps *= self.num_marks  # such treasure places
        self.max_steps += 25  # additional steps to come at treasure places

        marks = []

        for mark in range(self.num_marks):

            x, y = 0, 0
            while True:
                x, y = randint(3, len(self.cells[0]) - 4), randint(3, len(self.cells) - 4)

                found = True
                for xy in marks:
                    if (xy[0] - 4) < x < (xy[0] + 4) and (xy[1] - 4) <= y <= (xy[1] + 4):
                        found = False
                        break

                if not found:
                    continue

                marks.append((x, y))
                break

            for _x in range(3):
                for _y in range(3):
                    self.cells[y - 1 + _y][x - 1 + _x].type = self.Cell.Type.TREASURE
                    self.cells[y - 1 + _y][x - 1 + _x].value = 0.5
                    self.cells[y - 1 + _y][x - 1 + _x].treasure_residue = randint(self.TREASURE_MIN, self.TREASURE_MAX)
            self.cells[y][x].treasure_residue = self.TREASURE_MAX
            self.cells[y][x].value = 1

        return self._make_observation(False, 0, 0)

    def render(self):
        self.clock.tick(self.frame_rate)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        self.screen.fill(self.WHITE)

        for x in range(len(self.cells[0])):
            for y in range(len(self.cells)):
                cell = (x * self.cell_size[0], y * self.cell_size[1], self.cell_size[0], self.cell_size[1])

                if self.cells[y][x].type == self.Cell.Type.TREASURE:
                    component = int(255 * (1 - self.cells[y][x].treasure_residue / self.TREASURE_MAX))
                    # BLUE LINEAR GRADIENT (255, 0, 0), ..., (255, 255, 255) REGARDING TREASURE RESIDUE
                    color = (component, component, 255)
                    pygame.draw.rect(self.screen, color, cell)

                pygame.draw.rect(self.screen, self.BLACK, cell, 1)

        entity_cell = (self.entity.x * self.cell_size[0] + (self.cell_size[0] - self.entity_size[0]) / 2,
                       self.entity.y * self.cell_size[1] + (self.cell_size[1] - self.entity_size[1]) / 2,
                       self.entity_size[0], self.entity_size[1])
        pygame.draw.rect(self.screen, self.RED, entity_cell)

        pygame.display.set_caption("Steps remains: " + str(self.max_steps))

        pygame.display.update()

    def step(self, action):

        reward = 0
        value = 0

        if action == self.Entity.Actions.UP:
            if self.entity.y - 1 >= 0:
                self.entity.y -= 1
        elif action == self.Entity.Actions.RIGHT:
            if self.entity.x + 1 < len(self.cells[0]):
                self.entity.x += 1
        elif action == self.Entity.Actions.DOWN:
            if self.entity.y + 1 < len(self.cells):
                self.entity.y += 1
        elif action == self.Entity.Actions.LEFT:
            if self.entity.x - 1 >= 0:
                self.entity.x -= 1
        elif action == self.Entity.Actions.DIG:
            if self.cells[self.entity.y][self.entity.x].treasure_residue > 0:
                self.cells[self.entity.y][self.entity.x].treasure_residue -= 1
                reward = 1
                value = self.cells[self.entity.y][self.entity.x].value

        self.max_steps -= 1
        done = True if self.max_steps == 0 else False

        return self._make_observation(done, reward, value)

    def _make_observation(self, done, reward, value):
        shape = (len(self.cells), len(self.cells[0]))

        cells_type = np.empty(shape)
        treasure_map = np.empty(shape)
        value_map = np.empty(shape)
        for y in range(shape[0]):
            for x in range(shape[1]):
                cells_type[y][x] = self.cells[y][x].type.value
                treasure_map[y][x] = self.cells[y][x].treasure_residue
                value_map[y][x] = self.cells[y][x].value

        entity_map = np.zeros(shape)
        entity_map[self.entity.y][self.entity.x] = self.entity.coalition

        return self.Observation(done, reward, value, cells_type, entity_map, treasure_map, value_map)


if __name__ == '__main__':
    env = TIenv(frame_rate=10)
    obs = env.reset()
    print(obs)

    while True:
        env.render()
        obs = env.step(randint(0, len(env.Entity.Actions)))  # TIenv.Entity.Actions.RIGHT

        if obs.done:
            env.reset()
