import time
import selenium
from selenium import webdriver
from selenium.webdriver.common.by import By


class ReactionTimeEnvironment:
    def __init__(self):
        self.driver = webdriver.Chrome()

        self.driver.get("https://humanbenchmark.com/tests/reactiontime")

        # Await cookies popup
        time.sleep(3)

        cookies_accept = self.driver.find_element(By.CSS_SELECTOR, "button.css-47sehv")
        cookies_accept.click()

    def reset(self):
        self.driver.refresh()


reaction_time_env = ReactionTimeEnvironment()

input("Press Enter to quit...")

