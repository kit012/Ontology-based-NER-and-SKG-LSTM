from selenium import webdriver
from selenium.webdriver.chrome.options import Options

from stock_prediction.db_model.wc_yahoo_finance import WCYahooFinance
from stock_prediction.web_crawling.util import change_post_time_tz

from datetime import datetime
from bs4 import BeautifulSoup


def crawl_marketwatch(config, id, link):
    chrome_options = Options()
    chrome_driver_path = config.chrome_driver_path

    driver = webdriver.Chrome(chrome_driver_path, chrome_options=chrome_options)
    driver.get(link)

    main_content = driver.find_element_by_xpath(
        "//html/body/section[@class='container container--body']/div[@class='region region--primary']");

    header = main_content.find_element_by_xpath(
        "div[@class='column column--full article__header']/div[@class='article__masthead']");

    post_time_tag = header.find_element_by_xpath("time[@class='timestamp timestamp--pub']");

    post_time = datetime.strptime(
        post_time_tag.text.replace('p.m.', 'pm').replace('a.m.', 'am').replace('Sept', 'Sep').replace('First ', ''),
        'Published: %b. %d, %Y at %I:%M %p ET')
    post_time = change_post_time_tz(post_time)

    author = header.find_element_by_xpath("div[@class='byline article__byline']/div[@class='author  hasMenu']/h4").text;

    article = BeautifulSoup(main_content.find_element_by_xpath(
        "div[@class='column column--full article__content']/div[@class='article__body article-wrap at16-col16 barrons-article-wrap']").get_attribute(
        'innerHTML'), 'html.parser')

    article_text_list = []

    for x in article.find_all('p'):
        if x.find("strong") is None:
            article_text_list.append(x.text)

    article_text = ''.join(article_text_list)

    WCYahooFinance().update_news_article_by_id(id, post_time, author, article_text)
