import time
from selenium import webdriver
from selenium.webdriver.common.by import By

driver = webdriver.Chrome()

driver.get("https://humanbenchmark.com/tests/reactiontime")

# Await cookies popup
time.sleep(3)

cookies_accept = driver.find_element(By.CSS_SELECTOR, "button.css-47sehv")
cookies_accept.click()



input("Press Enter to quit...")

