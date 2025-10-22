
from playwright.sync_api import sync_playwright

def run():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto("http://localhost:8550")
        page.wait_for_selector("text=gondolkodik...", state="detached", timeout=30000)
        page.fill("input[type='text']", 'Create a diagram with "a -> b"')
        page.press("input[type='text']", "Enter")
        page.wait_for_selector("img", timeout=30000)
        page.screenshot(path="jules-scratch/verification/verification.png")
        browser.close()

if __name__ == "__main__":
    run()
