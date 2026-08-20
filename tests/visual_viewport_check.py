"""Capture the live Executive Overview at three responsive viewports.

Run Streamlit on localhost:8501 before executing this script.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.edge.options import Options
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import TimeoutException


ROOT = Path(__file__).resolve().parents[1]
BASE_URL = os.getenv("STREAMLIT_TEST_URL", "http://localhost:8501").rstrip("/")
SCREENSHOT_SET = os.getenv("STREAMLIT_SCREENSHOT_SET", "screenshots")
LOAD_TIMEOUT = int(os.getenv("STREAMLIT_LOAD_TIMEOUT", "180"))
OUTPUT = ROOT / "artifacts" / "testing" / SCREENSHOT_SET
VIEWPORTS = {
    "desktop-1440x1000": (1440, 1000),
    "tablet-1024x900": (1024, 900),
    "mobile-390x844": (390, 844),
}


def enter_app_context(driver: webdriver.Edge) -> None:
    """Enter Community Cloud's app iframe while remaining local-run compatible."""
    WebDriverWait(driver, LOAD_TIMEOUT).until(
        EC.title_contains("Customer Churn Intelligence")
    )
    frames = driver.find_elements(By.TAG_NAME, "iframe")
    if frames:
        driver.switch_to.frame(frames[0])


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-first-run")
    driver = webdriver.Edge(options=options)
    results = []
    try:
        for name, (width, height) in VIEWPORTS.items():
            driver.set_window_size(width, height)
            started = time.perf_counter()
            driver.get(BASE_URL)
            try:
                enter_app_context(driver)
                WebDriverWait(driver, LOAD_TIMEOUT).until(
                    lambda browser: len(browser.find_elements(By.CSS_SELECTOR, '[data-testid="stMetric"]')) >= 5
                )
            except TimeoutException:
                diagnostic = OUTPUT / f"{name}-load-timeout.png"
                driver.save_screenshot(str(diagnostic))
                raise TimeoutError(
                    f"Dashboard did not initialize at {BASE_URL}. "
                    f"Page title: {driver.title!r}. "
                    f"Screenshot: {diagnostic.relative_to(ROOT)}"
                )
            load_seconds = round(time.perf_counter() - started, 2)
            time.sleep(2)
            overflow = bool(driver.execute_script(
                "return document.documentElement.scrollWidth > document.documentElement.clientWidth"
            ))
            screenshot = OUTPUT / f"{name}.png"
            driver.save_screenshot(str(screenshot))
            results.append({
                "viewport": name,
                "theme": "light",
                "width": width,
                "height": height,
                "horizontal_overflow": overflow,
                "load_seconds": load_seconds,
                "screenshot": str(screenshot.relative_to(ROOT)),
            })
        driver.set_window_size(1440, 1000)
        started = time.perf_counter()
        driver.get(BASE_URL)
        enter_app_context(driver)
        WebDriverWait(driver, LOAD_TIMEOUT).until(
            lambda browser: len(browser.find_elements(By.CSS_SELECTOR, '[data-testid="stMetric"]')) >= 5
        )
        theme_toggle = WebDriverWait(driver, 30).until(
            EC.element_to_be_clickable((By.XPATH, "//label[contains(., 'Dark theme')]") )
        )
        theme_toggle.click()
        WebDriverWait(driver, 30).until(
            lambda browser: browser.execute_script(
                "return getComputedStyle(document.documentElement).colorScheme"
            ) == "dark"
        )
        time.sleep(2)
        screenshot = OUTPUT / "dark-desktop-1440x1000.png"
        driver.save_screenshot(str(screenshot))
        results.append({
            "viewport": "dark-desktop-1440x1000",
            "theme": "dark",
            "width": 1440,
            "height": 1000,
            "horizontal_overflow": bool(driver.execute_script(
                "return document.documentElement.scrollWidth > document.documentElement.clientWidth"
            )),
            "load_seconds": round(time.perf_counter() - started, 2),
            "screenshot": str(screenshot.relative_to(ROOT)),
        })
        for page, slug in [
            ("Customer Risk Predictor", "predictor"),
            ("Geographic Analysis", "geography"),
            ("Retention Simulator", "retention-simulator"),
        ]:
            navigation = WebDriverWait(driver, 30).until(
                EC.element_to_be_clickable((By.XPATH, f"//label[contains(., '{page}')]") )
            )
            navigation.click()
            time.sleep(8)
            if page == "Geographic Analysis":
                WebDriverWait(driver, 60).until(
                    lambda browser: len(browser.find_elements(By.CSS_SELECTOR, '[data-testid="stDataFrame"]')) == 1
                )
                time.sleep(8)
            else:
                time.sleep(2)
            screenshot = OUTPUT / f"dark-{slug}-1440x1000.png"
            driver.save_screenshot(str(screenshot))
            results.append({
                "viewport": f"dark-{slug}-1440x1000",
                "theme": "dark",
                "width": 1440,
                "height": 1000,
                "horizontal_overflow": bool(driver.execute_script(
                    "return document.documentElement.scrollWidth > document.documentElement.clientWidth"
                )),
                "screenshot": str(screenshot.relative_to(ROOT)),
            })
    finally:
        driver.quit()
    report_name = "viewport_results.json" if SCREENSHOT_SET == "screenshots" else f"{SCREENSHOT_SET}_results.json"
    report_path = ROOT / "artifacts" / "testing" / report_name
    report_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    if any(result["horizontal_overflow"] for result in results):
        raise AssertionError("Horizontal overflow detected in one or more viewports.")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
