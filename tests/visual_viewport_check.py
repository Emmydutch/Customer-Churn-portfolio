"""Capture the live Executive Overview at three responsive viewports.

Run Streamlit on localhost:8501 before executing this script.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.edge.options import Options
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts" / "testing" / "screenshots"
VIEWPORTS = {
    "desktop-1440x1000": (1440, 1000),
    "tablet-1024x900": (1024, 900),
    "mobile-390x844": (390, 844),
}


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
            driver.get("http://localhost:8501")
            WebDriverWait(driver, 60).until(
                EC.visibility_of_element_located((By.CLASS_NAME, "brand-title"))
            )
            WebDriverWait(driver, 60).until(
                lambda browser: len(browser.find_elements(By.CSS_SELECTOR, '[data-testid="stMetric"]')) >= 5
            )
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
                "screenshot": str(screenshot.relative_to(ROOT)),
            })
        driver.set_window_size(1440, 1000)
        driver.get("http://localhost:8501")
        WebDriverWait(driver, 60).until(
            EC.visibility_of_element_located((By.CLASS_NAME, "brand-title"))
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
            WebDriverWait(driver, 60).until(
                lambda browser: browser.find_element(By.CLASS_NAME, "brand-title").text == page
            )
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
    report_path = ROOT / "artifacts" / "testing" / "viewport_results.json"
    report_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    if any(result["horizontal_overflow"] for result in results):
        raise AssertionError("Horizontal overflow detected in one or more viewports.")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
