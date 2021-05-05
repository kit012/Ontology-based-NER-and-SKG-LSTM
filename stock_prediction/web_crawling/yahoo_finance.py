from selenium import webdriver
from selenium.webdriver.chrome.options import Options
# from selenium.webdriver.common.action_chains import ActionChains
from datetime import datetime
from stock_prediction.db_model.wc_yahoo_finance import WCYahooFinance
from stock_prediction.web_crawling.util import page_down, get_content_class_name

import time
from bs4 import BeautifulSoup
import re


def crawl_yahoo_finance(config):
    chrome_options = Options()
    chrome_driver_path = config.chrome_driver_path

    driver = webdriver.Chrome(chrome_driver_path, chrome_options=chrome_options)
    driver.get('https://finance.yahoo.com/')
    body = driver.find_element_by_css_selector('body');
    # Max: 170
    page_down(body, 5)
    time.sleep(3)

    main_content = driver.find_element_by_xpath("//html/body/div[1]/div/div/div[1]/div/div[3]/div[1]/div");
    news = main_content.find_element_by_css_selector("div[data-reactid='35']");

    news_li = news.find_elements_by_xpath("div/div/div/ul/li[@class='js-stream-content Pos(r)']");

    print(len(news_li))

    data = []

    update_time = datetime.now()

    for li in news_li:
        if li.find_element_by_xpath("div").get_attribute("class") not in (
                'controller gemini-ad native-ad-item Feedback Pos(r)', 'controller gemini-ad Feedback Pos(r)'):
            class_name = li.find_elements_by_xpath("div/div/div");

            news_box = BeautifulSoup(
                li.find_element_by_xpath(
                    "div/div/div[@class='" + get_content_class_name(class_name) + "']").get_attribute(
                    'innerHTML'), 'html.parser')

            category = news_box.find("div", {"data-test-locator": "catlabel"}).text
            source_span = news_box.find('div', class_=re.compile("^C\(#959595\)")).find_all('span')
            source = source_span[0].text
            # last = source_span[1].text
            title = news_box.h3.a.text
            link = news_box.h3.find('a', href=True)['href']
            summary = news_box.p.text

            result = WCYahooFinance().check_record_exist(link)

            if [x[0] for x in result][0] == 0:
                data.append({
                    'UP_DT': datetime.now(),
                    'CATEGORY': category,
                    'SRC': source,
                    'LINK': link,
                    'TITLE': title,
                    'SUMMARY': summary,
                })

            # print('category: %s' % category)
            # print('source: %s' % source)
            # print(last)
            # print(title)
            # print(link)
            # print(summary)

    driver.close()
    if len(data) > 0:
        WCYahooFinance().insert_new_records_batch(data)
