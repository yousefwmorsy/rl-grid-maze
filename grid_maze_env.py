from enum import Enum

import numpy as np
import pygame

import gymnasium as gym
from gymnasium import spaces


class Actions(Enum):
    RIGHT = 0
    UP = 1
    LEFT = 2
    DOWN = 3


class GridMazeEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=5):
        self.size = size  # The size of the square grid
        self.num_states = size * size
        self.window_size = 512  # The size of the PyGame window

        # Observations include the agent, target, and two bad-cell locations.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "target": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "bad_cells": spaces.Box(0, size - 1, shape=(2, 2), dtype=int),
            }
        )
        self._agent_location = np.array([-1, -1], dtype=int)
        self._start_location = np.array([-1, -1], dtype=int)
        self._target_location = np.array([-1, -1], dtype=int)
        self._bad_cells = np.empty((2, 2), dtype=int)

        # We have 4 actions, corresponding to "right", "up", "left", "down"
        self.action_space = spaces.Discrete(4)

        """
        The following dictionary maps abstract actions from `self.action_space` to
        the direction we will walk in if that action is taken.
        i.e. 0 corresponds to "right", 1 to "up" etc.
        Uses NumPy [row, col] convention where row 0 is at the top.
        """
        self._action_to_direction = {
            Actions.RIGHT.value: np.array([0, 1]),
            Actions.UP.value: np.array([-1, 0]),
            Actions.LEFT.value: np.array([0, -1]),
            Actions.DOWN.value: np.array([1, 0]),
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def state_from_location(self, location):
        """Encode an agent [row, column] location as a tabular state."""
        return int(location[0] * self.size + location[1])

    def location_from_state(self, state):
        """Decode a tabular state into an agent [row, column] location."""
        return np.array(divmod(state, self.size), dtype=int)

    def transition(self, state, action):
        """Return the deterministic transition for a tabular state and action."""
        location = self.location_from_state(state)
        target = self.state_from_location(self._target_location)
        if state == target:
            return state, 0, True

        next_location = np.clip(
            location + self._action_to_direction[action], 0, self.size - 1
        )
        next_state = self.state_from_location(next_location)
        terminated = next_state == target
        bad_cell_hit = any(
            np.array_equal(next_location, bad_cell) for bad_cell in self._bad_cells
        )
        reward = 1 if terminated else -1 if bad_cell_hit else 0
        return next_state, reward, terminated

    def _get_obs(self):
        return {
            "agent": self._agent_location.copy(),
            "target": self._target_location.copy(),
            "bad_cells": self._bad_cells.copy(),
        }

    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the agent's location uniformly at random
        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)
        self._start_location = self._agent_location.copy()

        # We will sample the target's location randomly until it does not coincide with the agent's location
        self._target_location = self._agent_location
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = self.np_random.integers(
                0, self.size, size=2, dtype=int
            )

        # Sample two distinct bad cells that do not overlap the goal.
        cells = np.array(
            [(row, column) for row in range(self.size) for column in range(self.size)],
            dtype=int,
        )
        valid_cells = cells[
            ~np.all(cells == self._target_location, axis=1)
            & ~np.all(cells == self._start_location, axis=1)
        ]

        # If the goal is in a corner, do not block both of its only exits.
        corner_exits = []
        row, column = self._target_location
        if row in (0, self.size - 1) and column in (0, self.size - 1):
            corner_exits = [
                (1 if row == 0 else self.size - 2, column),
                (row, 1 if column == 0 else self.size - 2),
            ]

        while True:
            bad_cells = valid_cells[
                self.np_random.choice(len(valid_cells), size=2, replace=False)
            ]
            if not corner_exits or not all(
                any(np.array_equal(cell, exit_cell) for cell in bad_cells)
                for exit_cell in corner_exits
            ):
                self._bad_cells = bad_cells
                break

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self._action_to_direction[action]
        # We use `np.clip` to make sure we don't leave the grid
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )
        # An episode is done iff the agent has reached the target
        terminated = np.array_equal(self._agent_location, self._target_location)
        bad_cell_hit = any(
            np.array_equal(self._agent_location, bad_cell) for bad_cell in self._bad_cells
        )
        reward = 1 if terminated else -1 if bad_cell_hit else 0 
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def render(self):
        return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # Draw the start, target, and bad cells with their labels.
        cell_markers = [
            (self._start_location, (255, 255, 0), "S"),
            (self._target_location, (0, 200, 0), "G"),
        ]
        for bad_cell in self._bad_cells:
            cell_markers.append((bad_cell, (255, 0, 0), "X"))

        for location, color, label in cell_markers:
            cell_rect = pygame.Rect(
                pix_square_size * location[::-1],
                (pix_square_size, pix_square_size),
            )
            pygame.draw.rect(canvas, color, cell_rect)
            glyphs = {
                "S": (
                    "11111",
                    "10000",
                    "10000",
                    "11111",
                    "00001",
                    "00001",
                    "11111",
                ),
                "G": (
                    "11111",
                    "10000",
                    "10000",
                    "10111",
                    "10001",
                    "10001",
                    "11111",
                ),
                "X": (
                    "10001",
                    "01010",
                    "00100",
                    "00100",
                    "01010",
                    "10001",
                    "10001",
                ),
            }
            glyph = glyphs[label]
            pixel_size = min(cell_rect.width, cell_rect.height) / 9
            glyph_width = pixel_size * 5
            glyph_height = pixel_size * 7
            glyph_left = cell_rect.centerx - glyph_width / 2
            glyph_top = cell_rect.centery - glyph_height / 2
            for row, glyph_row in enumerate(glyph):
                for column, pixel in enumerate(glyph_row):
                    if pixel == "1":
                        pygame.draw.rect(
                            canvas,
                            (0, 0, 0),
                            pygame.Rect(
                                glyph_left + column * pixel_size,
                                glyph_top + row * pixel_size,
                                pixel_size,
                                pixel_size,
                            ),
                        )

        # Keep the current agent visible without hiding the cell labels.
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location[::-1] + 0.5) * pix_square_size,
            pix_square_size / 3,
            width=4,
        )

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            self.clock.tick(self.metadata["render_fps"])
        elif self.render_mode == "rgb_array":
            return np.transpose(
                pygame.surfarray.array3d(canvas),
                axes=(1, 0, 2),
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
            self.clock = None