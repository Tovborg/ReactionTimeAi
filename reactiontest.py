import time
import re
import selenium
from selenium import webdriver
from selenium.webdriver.common.by import By


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
        game_started = False
        game_finished = False
        early_click = False

        try:
            self.driver.find_element(By.CSS_SELECTOR, "div.view-splash")
            game_started = False
        except selenium.common.exceptions.NoSuchElementException:
            game_started = True

        try:
            self.driver.find_element(By.CSS_SELECTOR, "div.view-result")
            game_finished = True
            self.reset()
        except selenium.common.exceptions.NoSuchElementException:
            pass  # Game over indicator not found, game is still ongoing

        try:
            self.driver.find_element(By.CSS_SELECTOR, "div.view-scold")
            early_click = True
            self.reset()
        except selenium.common.exceptions.NoSuchElementException:
            pass  # Early click indicator not found, click was not too early

        state = {
            "square_color": square_color,
            "game_started": game_started,
            "game_finished": game_finished,
            "early_click": early_click
        }
        print(state)
        return state

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
        if action == "click":
            self.click_square()

        state = self.get_state()

        reward, done = self.calculate_reward_and_done(state, action=action)
        print(reward, done)

    def calculate_reward_and_done(self, state, action):
        reward = 0
        done = False

        if state['game_started'] is False and state['square_color'] == "#2b87d1" and action == "click":
            reward += 100
            done = False
        if state['game_started'] and state['game_finished'] is False and action == "click" and state['square_color'] == "#4bdb6a":
            reward += 100
            done = True

            agent_reaction_time = self.get_reaction_time()
            if agent_reaction_time < self.best_reaction_time:
                self.best_reaction_time = agent_reaction_time
                reward += 150
            else:
                reward -= 150
        if state['game_started'] and state['game_finished'] is False and action == "click" and state['early_click']:
            reward -= 200
            done = True

        return reward, done



