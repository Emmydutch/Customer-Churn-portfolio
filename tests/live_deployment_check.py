"""Exercise the public Community Cloud deployment in a real browser."""

from __future__ import annotations

import json
import os
import time

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.edge.options import Options
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait


BASE_URL = os.environ["STREAMLIT_TEST_URL"].rstrip("/")
TIMEOUT = int(os.getenv("STREAMLIT_LOAD_TIMEOUT", "120"))
PAGES = [
    "Executive Overview",
    "Customer Analysis",
    "Churn Drivers",
    "Geographic Analysis",
    "Customer Risk Predictor",
    "Retention Simulator",
    "Methodology",
]


def main() -> None:
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-first-run")
    driver = webdriver.Edge(options=options)
    driver.set_window_size(1440, 1000)
    results: dict[str, object] = {"url": BASE_URL, "pages": {}}
    try:
        started = time.perf_counter()
        driver.get(BASE_URL)
        frame = WebDriverWait(driver, TIMEOUT).until(
            lambda browser: browser.find_elements(By.TAG_NAME, "iframe")[0]
            if browser.find_elements(By.TAG_NAME, "iframe")
            else False
        )
        driver.switch_to.frame(frame)
        WebDriverWait(driver, TIMEOUT).until(
            lambda browser: len(browser.find_elements(By.CSS_SELECTOR, '[data-testid="stMetric"]')) >= 5
        )
        results["initial_load_seconds"] = round(time.perf_counter() - started, 2)
        results["pages"]["Executive Overview"] = "passed"

        for page in PAGES[1:]:
            navigation = WebDriverWait(driver, 30).until(
                EC.element_to_be_clickable((By.XPATH, f"//label[contains(., '{page}')]"))
            )
            navigation.click()
            time.sleep(8)
            results["pages"][page] = "passed"

            if page == "Geographic Analysis":
                WebDriverWait(driver, TIMEOUT).until(
                    lambda browser: len(browser.find_elements(By.CSS_SELECTOR, '[data-testid="stDataFrame"]')) == 1
                )

            if page == "Customer Risk Predictor":
                submit = WebDriverWait(driver, 30).until(
                    EC.element_to_be_clickable((By.XPATH, "//button[contains(., 'Calculate churn risk')]"))
                )
                driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", submit)
                submit.click()
                WebDriverWait(driver, TIMEOUT).until(
                    EC.presence_of_element_located((By.XPATH, "//*[contains(., 'Predicted churn probability')]"))
                )
                results["predictor_submission"] = "passed"

            if page == "Retention Simulator":
                metrics = driver.find_elements(By.CSS_SELECTOR, '[data-testid="stMetric"]')
                before = [metric.text for metric in metrics]
                slider = WebDriverWait(driver, 30).until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, '[data-testid="stSlider"] [role="slider"]'))
                )
                slider.send_keys(Keys.ARROW_RIGHT)
                WebDriverWait(driver, TIMEOUT).until(
                    lambda browser: [
                        metric.text for metric in browser.find_elements(By.CSS_SELECTOR, '[data-testid="stMetric"]')
                    ] != before
                )
                results["retention_interaction"] = "passed"

        results["status"] = "passed"
        print(json.dumps(results, indent=2))
    finally:
        driver.quit()


if __name__ == "__main__":
    main()
