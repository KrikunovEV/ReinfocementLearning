import pygame
import sys
import numpy as np
from torch.nn.functional import softmax

from enum import IntEnum, Enum
from random import randint
from dataclasses import dataclass


class TIenv:

    @dataclass
    class Observation:

        @dataclass
        class Feature:

            class Type(Enum):
                SCALAR = 0,
                CATEGORICAL = 1

            data:  np.ndarray
            type:  Type
            scale: int

        done:   bool
        reward: float
        available_actions: list
        cell_map:     Feature
        entity_map:   Feature
        treasure_map: Feature

        def __init__(self, done, reward, shape, available_actions):
            self.done = done
            self.reward = reward
            self.available_actions = available_actions
            self.cell_map = self.Feature(np.empty(shape), self.Feature.Type.CATEGORICAL, 2)
            self.entity_map = self.Feature(np.zeros(shape), self.Feature.Type.CATEGORICAL, 2)
            self.treasure_map = self.Feature(np.empty(shape), self.Feature.Type.SCALAR, 0)

    class Cell:

        class Type(Enum):
            EMPTY = 0
            TREASURE = 1

        treasure_residue = 0
        type = Type.EMPTY

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

        def available_actions(self, feature_size):
            actions = [a.value for a in self.Actions]

            if self.x == 0:
                actions.remove(self.Actions.LEFT)
            if self.y == 0:
                actions.remove(self.Actions.UP)
            if self.x == feature_size[1] - 1:
                actions.remove(self.Actions.RIGHT)
            if self.y == feature_size[0] - 1:
                actions.remove(self.Actions.DOWN)

            return actions

    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    RED = (255, 0, 0)

    TREASURE_MAX = 25
    TREASURE_MIN = 5

    def __init__(self, frame_rate=0, num_marks=2, feature_size=(16, 16)):
        self.frame_rate = frame_rate
        self.num_marks = num_marks
        self.feature_size = feature_size

        pygame.init()

        screen_height = pygame.display.Info().current_h - 150
        self.window_size = (int(screen_height * feature_size[1] / feature_size[0]), screen_height)
        self.cell_size = (int(self.window_size[0] / feature_size[1]), int(self.window_size[1] / feature_size[0]))
        self.entity_size = (int(self.cell_size[0] * 0.8), int(self.cell_size[1] * 0.8))

        self.clock = pygame.time.Clock()
        self.screen = pygame.display.set_mode(self.window_size)

    def __del__(self):
        pygame.quit()

    def reset(self):
        self.cells = [[self.Cell() for x in range(self.feature_size[1])] for y in range(self.feature_size[0])]

        self.entity = self.Entity()
        self.entity.x = np.random.randint(0, self.feature_size[1])
        self.entity.y = np.random.randint(0, self.feature_size[0])

        self.max_steps = (self.TREASURE_MAX + self.TREASURE_MIN) / 2  # mean (expected) treasure count in one cell
        self.max_steps *= 9  # such cells is 9 in one treasure place
        self.max_steps *= self.num_marks  # such treasure places
        self.max_steps += sum(self.feature_size)  # additional steps

        lim = self.feature_size[1] / self.num_marks
        for mark in range(self.num_marks):
            x = randint(lim * mark + 1, lim * (mark + 1) - 2)
            y = randint(1, self.feature_size[0] - 2)

            for _x in range(3):
                for _y in range(3):
                    y_index = y - 1 + _y
                    x_index = x - 1 + _x
                    self.cells[y_index][x_index].type = self.Cell.Type.TREASURE
                    self.cells[y_index][x_index].treasure_residue = randint(self.TREASURE_MIN, self.TREASURE_MAX)
            self.cells[y][x].treasure_residue = self.TREASURE_MAX

        return self._make_observation(False, 0)

    def render(self, draw_grid=True):
        if self.frame_rate != 0:
            self.clock.tick(self.frame_rate)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        self.screen.fill(self.WHITE)

        for y in range(self.feature_size[0]):
            for x in range(self.feature_size[1]):
                cell = (x * self.cell_size[0], y * self.cell_size[1], self.cell_size[0], self.cell_size[1])

                if self.cells[y][x].type == self.Cell.Type.TREASURE:
                    component = int(255 * (1 - self.cells[y][x].treasure_residue / self.TREASURE_MAX))
                    color = (component, component, 255)
                    pygame.draw.rect(self.screen, color, cell)

                if draw_grid:
                    pygame.draw.rect(self.screen, self.BLACK, cell, 1)

        entity_cell = (self.entity.x * self.cell_size[0] + (self.cell_size[0] - self.entity_size[0]) / 2,
                       self.entity.y * self.cell_size[1] + (self.cell_size[1] - self.entity_size[1]) / 2,
                       self.entity_size[0], self.entity_size[1])
        pygame.draw.rect(self.screen, self.RED, entity_cell)

        pygame.display.set_caption("Steps remain: " + str(self.max_steps))

        pygame.display.update()

    def step(self, action):
        reward = 0

        if action not in self.entity.available_actions(self.feature_size):
            raise ValueError('Action {action_id} is not available'.format(action_id=action))
        elif action == self.Entity.Actions.UP:
            self.entity.y -= 1
        elif action == self.Entity.Actions.RIGHT:
            self.entity.x += 1
        elif action == self.Entity.Actions.DOWN:
            self.entity.y += 1
        elif action == self.Entity.Actions.LEFT:
            self.entity.x -= 1
        elif action == self.Entity.Actions.DIG:
            x, y = self.entity.x, self.entity.y
            if self.cells[y][x].type == self.Cell.Type.TREASURE:
                self.cells[y][x].treasure_residue -= 1
                if self.cells[y][x].treasure_residue == 0:
                    self.cells[y][x].type = self.Cell.Type.EMPTY
                reward = 1

        self.max_steps -= 1
        done = True if self.max_steps == 0 else False

        return self._make_observation(done, reward)

    def _make_observation(self, done, reward):
        observation = self.Observation(done, reward, self.feature_size, self.entity.available_actions(self.feature_size))

        for y in range(self.feature_size[0]):
            for x in range(self.feature_size[1]):
                observation.cell_map.data[y][x] = self.cells[y][x].type.value
                observation.treasure_map.data[y][x] = self.cells[y][x].treasure_residue

        observation.entity_map.data[self.entity.y][self.entity.x] = self.entity.coalition

        return observation

    def save_value_and_policy_map_for_A2C(self, model, filename, obs=None):

        if obs is None:
            obs = self._make_observation(False, 0)

        obs.entity_map.data[self.entity.y][self.entity.x] = 0

        value_map = np.empty((self.feature_size[0], self.feature_size[1]))
        policy_map = np.empty((self.feature_size[0], self.feature_size[1]))

        for y in range(self.feature_size[0]):
            for x in range(self.feature_size[1]):
                obs.entity_map.data[y][x] = 1
                logits, value = model(obs)
                value_map[y][x] = value.item()
                obs.entity_map.data[y][x] = 0

                temp_entity = self.Entity(x, y)
                available_actions = temp_entity.available_actions(self.feature_size)
                policy = softmax(logits[available_actions], dim=-1)
                probabilities = policy.detach().numpy()
                probability = np.random.choice(probabilities, 1, p=probabilities)
                action = available_actions[np.where(probabilities == probability)[0][0]]
                policy_map[y][x] = action

        value_map -= np.min(value_map)
        value_map *= 255 / np.max(value_map)
        value_map = np.asarray(value_map, dtype=np.uint8)

        surface = pygame.Surface(self.window_size)
        surface.fill(self.WHITE)

        for y in range(self.feature_size[0]):
            for x in range(self.feature_size[1]):

                if self.cells[y][x].type == self.Cell.Type.TREASURE:
                    component = int(255 * (1 - self.cells[y][x].treasure_residue / self.TREASURE_MAX))
                    color = (component, component, 255)
                    cell = (x * self.cell_size[0], y * self.cell_size[1], self.cell_size[0], self.cell_size[1])
                    pygame.draw.rect(surface, color, cell)

                value_cell = (x * self.cell_size[0] + (self.cell_size[0] - self.entity_size[0]) / 2,
                              y * self.cell_size[1] + (self.cell_size[1] - self.entity_size[1]) / 2,
                              self.entity_size[0], self.entity_size[1])
                component = 255 - value_map[y][x]
                color = (component, component, component)
                pygame.draw.rect(surface, color, value_cell)

                start = (x * self.cell_size[0] + self.cell_size[0] / 2, y * self.cell_size[1] + self.cell_size[1] / 2)
                if policy_map[y][x] == self.Entity.Actions.UP:
                    end = (start[0], y * self.cell_size[1])
                    pygame.draw.line(surface, self.RED, start, end)
                elif policy_map[y][x] == self.Entity.Actions.DOWN:
                    end = (start[0], y * self.cell_size[1] + self.cell_size[1])
                    pygame.draw.line(surface, self.RED, start, end)
                elif policy_map[y][x] == self.Entity.Actions.LEFT:
                    end = (x * self.cell_size[0], start[1])
                    pygame.draw.line(surface, self.RED, start, end)
                elif policy_map[y][x] == self.Entity.Actions.RIGHT:
                    end = (x * self.cell_size[0] + self.cell_size[0], start[1])
                    pygame.draw.line(surface, self.RED, start, end)
                elif policy_map[y][x] == self.Entity.Actions.DIG:
                    start = (x * self.cell_size[0], y * self.cell_size[1])
                    end = (start[0] + self.cell_size[0], start[1] + self.cell_size[1])
                    pygame.draw.line(surface, self.RED, start, end)
                    start = (x * self.cell_size[0] + self.cell_size[0], y * self.cell_size[1])
                    end = (x * self.cell_size[0], y * self.cell_size[1] + self.cell_size[1])
                    pygame.draw.line(surface, self.RED, start, end)

        pygame.image.save(surface, filename)


if __name__ == '__main__':
    env = TIenv(frame_rate=30)
    obs = env.reset()
    print(obs)

    while True:
        env.render()
        obs = env.step(randint(0, len(env.Entity.Actions)))  # TIenv.Entity.Actions.RIGHT

        if obs.done:
            env.reset()
