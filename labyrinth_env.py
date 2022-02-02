from __future__ import print_function

import gym
from gym import spaces
import numpy as np
import random

actions = {"U": 0, "D": 1, "L": 2, "R": 3}

actionFromId = ["UP", "DOWN", "LEFT", "RIGHT"]

cells = {"empty": 0, "wall": 1, "agent": 2, "entrance": 3, "exit": 4}

chars = {
    "empty": "   ",
    "wall": " ■ ",
    "agent": " A ",
    "entrance": " E ",
    "exit": " U ",
}


class LabyrinthEnv(gym.Env):
    def __init__(self, max_actions, grid_h=0, grid_w=0, wall_percentage=0, load=False):
        """
        Initialized the maze environment
            grid_h (int) height of the maze
            grid_w (int) width of the maze
            wall_percentage (int) percentage of wall cells in the maze
            max_actions (int) max number of actions the agent can make
        """
        super(LabyrinthEnv, self).__init__()

        if load:
            self.load_labyrinth()
        else:
            self.labyrinth_size = {  # labyrinth exit (grid_h, grid_w)
                "h": grid_h + 2,
                "w": grid_w + 2,
                "p": wall_percentage,
            }
            """
            Labyrinth representation
                each cell is 
                    0 = empty
                    1 = wall
                    2 = agent
                    3 = entrance
                    4 = exit
            """
            self.labyrinth = self.generate_labyrinth()
            self.save_labyrinth()

        self.agent_position = {"x": 1, "y": 1}  # labyrinth entrance (1,1)
        self.max_actions = max_actions

        """
            4 possible actions
            Up Down Left Right
        """
        self.action_space = spaces.Discrete(4)
        self.possible_actions = [i for i in range(4)]

        """
            Agent neighborhood
            0 1 2 
            3 A 6
            7 8 9

            a list with 8 elements, with int values 0 or 1
            
            observation can have 2^8=256 possible states
        """
        self.state_space = [i for i in range(256)]

    def save_labyrinth(self):
        """
        Saves the labyrinth to a file as a matrix of int
        """
        np.savetxt("labyrinth", self.labyrinth, delimiter=" ", fmt="%d")

    def load_labyrinth(self):
        """
        Loads the labyrinth from a file
        """
        self.labyrinth = np.loadtxt("labyrinth", delimiter=" ", dtype=int)
        self.labyrinth_size = {"h": len(self.labyrinth), "w": len(self.labyrinth[0])}
        print(self.labyrinth_size)

    def render(self, mode="human"):
        labyrinth_h = self.labyrinth_size["h"]
        labyrinth_w = self.labyrinth_size["w"]

        labyrinth_with_info = self.labyrinth.copy()
        labyrinth_with_info[1][1] = 3  # entrance
        labyrinth_with_info[labyrinth_h - 2][labyrinth_w - 2] = 4  # exit
        labyrinth_with_info[self.agent_position["y"]][
            self.agent_position["x"]
        ] = 2  # agent

        for y in range(labyrinth_h):
            for x in range(labyrinth_w):
                cell = labyrinth_with_info[y][x]
                if cell == cells["wall"]:  # 1
                    print(chars["wall"], end="")
                elif cell == cells["agent"]:  # 2
                    print(chars["agent"], end="")
                elif cell == cells["entrance"]:  # 3
                    print(chars["entrance"], end="")
                elif cell == cells["exit"]:  # 4
                    print(chars["exit"], end="")
                else:  # 0
                    print(chars["empty"], end="")
            print("")

    def reset(self):
        self.agent_position = {"x": 1, "y": 1}
        _, observation = self.next_observation()
        return observation

    def state_to_int(self, state):
        """
        Transforms the state list into his binary representation
        Returns the int value of the binary string
        """
        return int("".join(map(str, state)), 2)

    def next_observation(self, action=None):
        agent_x = self.agent_position["x"]
        agent_y = self.agent_position["y"]
        bumped_wall = False
        if action == actions["U"]:  # 0
            if self.labyrinth[agent_y - 1][agent_x] == 1:
                bumped_wall = True
            else:
                agent_y -= 1
        elif action == actions["D"]:  # 1
            if self.labyrinth[agent_y + 1][agent_x] == 1:
                bumped_wall = True
            else:
                agent_y += 1
        elif action == actions["L"]:  # 2
            if self.labyrinth[agent_y][agent_x - 1] == 1:
                bumped_wall = True
            else:
                agent_x -= 1
        elif action == actions["R"]:  # 3
            if self.labyrinth[agent_y][agent_x + 1] == 1:
                bumped_wall = True
            else:
                agent_x += 1

        self.agent_position["x"] = agent_x
        self.agent_position["y"] = agent_y

        state = [
            self.labyrinth[agent_y - 1][agent_x - 1],  # 0
            self.labyrinth[agent_y - 1][agent_x],  # 1
            self.labyrinth[agent_y - 1][agent_x + 1],  # 2
            self.labyrinth[agent_y][agent_x - 1],  # 3
            # A, the agent is here
            self.labyrinth[agent_y][agent_x + 1],  # 4
            self.labyrinth[agent_y + 1][agent_x - 1],  # 5
            self.labyrinth[agent_y + 1][agent_x],  # 6
            self.labyrinth[agent_y + 1][agent_x + 1],  # 7
        ]

        return bumped_wall, self.state_to_int(state)

    def step(self, action: int):
        bumped_wall, observation = self.next_observation(action)
        reward = -1
        done = False

        if bumped_wall:
            reward = -5

        if (
            self.agent_position["x"] == self.labyrinth_size["w"] - 2
            and self.agent_position["y"] == self.labyrinth_size["h"] - 2
        ):
            done = True
            reward = 10

        self.max_actions = self.max_actions - 1
        if self.max_actions == 0:
            done = True

        return observation, reward, done, {}

    def generate_labyrinth(self):
        labyrinth_h = self.labyrinth_size["h"]
        labyrinth_w = self.labyrinth_size["w"]
        walls_to_insert = int(
            ((labyrinth_h - 2) * (labyrinth_w - 2) * self.labyrinth_size["p"]) / 100
        )

        labyrinth = np.zeros((labyrinth_h, labyrinth_w), dtype=int)

        for y in range(labyrinth_h):
            for x in range(labyrinth_w):
                if (
                    (y == 0)
                    or (y == labyrinth_h - 1)
                    or (x == 0)
                    or (x == labyrinth_w - 1)
                ):
                    labyrinth[y][x] = 1

        while walls_to_insert:
            x = random.randint(1, labyrinth_w - 2)
            y = random.randint(1, labyrinth_h - 2)
            if labyrinth[y][x] == 0:
                labyrinth[y][x] = 1
                walls_to_insert = walls_to_insert - 1
        return labyrinth

    def action_space_sample(self):
        return np.random.choice(self.possible_actions)
