import time
import re
import selenium
from selenium import webdriver
from selenium.webdriver.common.by import By
from collections import namedtuple, deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import math
from itertools import count
import numpy as np


class ReactionTimeEnvironment:
    def __init__(self):
        self.driver = webdriver.Chrome()
        # Initialize at extremely high value to ensure that the first reaction time is always better
        self.best_reaction_time = 1000000000
        self.game_started = False

        self.driver.get("https://humanbenchmark.com/tests/reactiontime")
        # Await cookies popup
        time.sleep(3)
        cookies_accept = self.driver.find_element(By.CSS_SELECTOR, "button.css-47sehv")
        cookies_accept.click()

    # Environment interaction methods
    def reset(self):
        self.driver.refresh()

    def click_square(self):
        square = self.driver.find_element(By.CSS_SELECTOR, "div.css-saet2v")
        square.click()

    # Environment state methods
    def get_state(self):
        square_color = self.get_square_color()

        # Check for different states
        game_started = 0  # 0 for False and 1 for True
        game_finished = 0  # 0 for False and 1 for True
        early_click = 0  # 0 for False and 1 for True

        try:
            self.driver.find_element(By.CSS_SELECTOR, "div.view-splash")
            game_started = 0
        except selenium.common.exceptions.NoSuchElementException:
            game_started = 1

        try:
            self.driver.find_element(By.CSS_SELECTOR, "div.view-result")
            game_finished = 1
            self.reset()
        except selenium.common.exceptions.NoSuchElementException:
            pass  # Game over indicator not found, game is still ongoing

        try:
            self.driver.find_element(By.CSS_SELECTOR, "div.view-scold")
            early_click = 1
            self.reset()
        except selenium.common.exceptions.NoSuchElementException:
            pass  # Early click indicator not found, click was not too early

        # Preprocess hex color
        hex_color = square_color.lstrip('#')
        numerical_color = int(hex_color, 16)
        state = [numerical_color, game_started, game_finished, early_click]
        return np.array(state)

    def get_square_color(self):
        square = self.driver.find_element(By.CSS_SELECTOR, "div.css-saet2v")
        square_color_hex = '#{:02x}{:02x}{:02x}'.format(*map(int, square.value_of_css_property("background-color").strip("rgba()").split(", ")))
        return square_color_hex

    def get_reaction_time(self):
        reaction_time = self.driver.find_element(By.CSS_SELECTOR, "div.css-1qvtbrk.e19owgy78").text
        react_time_number = int(re.search(r'\d+', reaction_time).group())

        print(f"Your reaction time is {react_time_number}ms")
        return react_time_number

    # Gym methods
    def step(self, action):
        if action == 0:
            self.click_square()

        state = self.get_state()

        reward, done = self.calculate_reward_and_done(state, action=action)
        print(f"Reward: {reward}, Done: {done}")
        return state, reward, done

    def calculate_reward_and_done(self, state, action):
        reward = 0
        done = False

        square_color, game_started, game_finished, early_click = state

        if not game_started and square_color == "#2b87d1" and action == 0:
            reward += 100
            done = False
        if game_started and not game_finished and action == 0 and square_color == "#4bdb6a":
            reward += 100
            done = True

            agent_reaction_time = self.get_reaction_time()
            if agent_reaction_time < self.best_reaction_time:
                self.best_reaction_time = agent_reaction_time
                reward += 50
            else:
                reward -= 50
        if game_started and not game_finished and action == 0 and early_click:
            reward -= 200
            done = True

        return reward, done


# Deep Q-Network
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# Define the neural network
class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


# We don't need a select action method as we only have one action 0
BATCH_SIZE = 64
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

env = ReactionTimeEnvironment()
n_actions = 1
n_observations = len(env.get_state())
state = env.get_state()

device = (
    torch.device("cuda") if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    torch.device("cpu")
)

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))
    print(f"batch.state: {batch.state}")
    print(f"batch.action: {batch.action}")
    print(f"batch.reward: {batch.reward}")
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                       if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch.unsqueeze(1))
    next_state_values = torch.zeros(BATCH_SIZE, device=device)

    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values

    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


num_episodes = 300

for i_episode in range(num_episodes):
    env.reset()
    state_np = env.get_state()
    state = torch.tensor(state_np, dtype=torch.float32, device=device).unsqueeze(0)
    for t in count():
        action = torch.tensor([0], device=device, dtype=torch.long)
        next_state, reward, done = env.step(action)
        reward = torch.tensor([reward], device=device)
        if done:
            next_state = None
        else:
            next_state = torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0)

        memory.push(state, action, next_state, reward)
        state = next_state
        optimize_model()

        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1 - TAU)
            target_net.load_state_dict(target_net_state_dict)
        if done:
            print(f"Episode {i_episode} finished")
            break

print("Training finished")



