import time
import re
import selenium
from selenium import webdriver
from selenium.webdriver.common.by import By


class ReactionTimeEnvironment:
    def __init__(self):
        self.driver = webdriver.Chrome()
        self.reaction_time = None
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
        # check for view-splash class
        try:
            view_splash = self.driver.find_element(By.CSS_SELECTOR, "div.view-splash")
            game_started = False
        except selenium.common.exceptions.NoSuchElementException:
            game_started = True
        try:
            game_over = self.driver.find_element(By.CSS_SELECTOR, "div.view-result")
            game_over = True
        except selenium.common.exceptions.NoSuchElementException:
            game_over = False

        state = {
            "square_color": square_color,
            "game_started": game_started,
            "game_over": game_over,
        }

        return state

    def get_square_color(self):
        square = self.driver.find_element(By.CSS_SELECTOR, "div.css-saet2v")
        square_color_hex = '#{:02x}{:02x}{:02x}'.format(*map(int, square.value_of_css_property("background-color").strip("rgba()").split(", ")))
        return square_color_hex

    def get_reaction_time(self):
        reaction_time = self.driver.find_element(By.CSS_SELECTOR, "div.css-1qvtbrk.e19owgy78").text
        react_time_number = int(re.search(r'\d+', reaction_time).group())
        self.reaction_time = react_time_number
        print(f"Your reaction time is {react_time_number}ms")

    # Gym methods
    def step(self, action):
        if action == "click":
            self.click_square()

        state = self.get_state()

    def calculate_reward_and_done(self):
        pass




reaction_time_env = ReactionTimeEnvironment()

while True:
    reaction_time_env.get_state()
    if reaction_time_env.get_state()["game_over"]:
        reaction_time_env.get_reaction_time()
        reaction_time_env.reset()
    time.sleep(0.2)



