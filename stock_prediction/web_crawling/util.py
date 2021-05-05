from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys

from stock_prediction.config import Config

import time
import re
import pytz

config = Config()


def page_down(stuff, times, by=4, pause_sec=3):
    for i in range(times):
        for j in range(by):
            stuff.send_keys(Keys.PAGE_DOWN);
        time.sleep(pause_sec)


def get_content_class_name(class_names):
    for name in class_names:
        name = name.get_attribute("class");
        if re.search("^Ov\(h\).+Pend\(44px\)", name):
            return name


def fix_month_format(x):
    try:
        month = re.search("\s.+\.\s[0-9]", x).group(0)
        month = month.strip().split('.')[0]
    except AttributeError:
        month = ''

    if len(month) == 4:
        return x.replace(month, month[:3])
    else:
        return x


def change_post_time_tz(post_time):
    hkt = pytz.timezone('Hongkong')
    eastern = pytz.timezone('US/Eastern')

    post_time_est = eastern.localize(post_time)

    return post_time_est.astimezone(hkt)


def get_chrome_driver():
    chrome_options = Options()
    # chrome_options.add_argument("--headless")
    # chrome_options.add_argument('--disable-gpu')

    chrome_driver_path = config.chrome_driver_path

    return webdriver.Chrome(chrome_driver_path, chrome_options=chrome_options)
