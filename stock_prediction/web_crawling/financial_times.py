from selenium.common.exceptions import NoSuchElementException

from stock_prediction.web_crawling.util import get_chrome_driver
from stock_prediction.db_model.proquest_ft import ProquestFt

# from getpass import getpass
from datetime import datetime
from bs4 import BeautifulSoup
import time


def crawl_ft(config):
    driver = get_chrome_driver()

    uid = config.ft_uid
    password = config.ft_password

    driver.get(
        'https://www-proquest-com.eproxy.lib.hku.hk/publication/35024?OpenUrlRefId=info:xri/sid:primo&accountid=14548')

    time.sleep(5)

    uid_textbox = driver.find_element_by_xpath('/html/body/main/div/form/div/div/div[1]/input');
    uid_textbox.send_keys(uid)

    password_textbox = driver.find_element_by_xpath('/html/body/main/div/form/div/div/div[2]/input');
    password_textbox.send_keys(password)
    password_textbox.submit()

    time.sleep(5)

    # driver.find_element_by_xpath(
    #     '/html/body/div[4]/div[1]/div/div[9]/div[2]/div[2]/div/div[2]/div[2]/div/div[1]/div[1]/div/div/div/div[2]/nav/ul/li[6]/a').click();
    #
    # time.sleep(1)
    #
    # driver.find_element_by_xpath(
    #     '/html/body/div[4]/div[1]/div/div[9]/div[2]/div[2]/div/div[2]/div[2]/div/div[1]/div[1]/div/div/div/div[2]/nav/ul/li[6]/a').click();
    #
    # time.sleep(1)

    # year = driver.find_element_by_xpath('/html/body/div[4]/div[1]/div/div[9]/div[2]/div[2]/div/div[2]/div[1]/div/div/div/form/div[3]/ul/li[1]/select');

    table = driver.find_elements_by_xpath(
        '/html/body/div[4]/div[1]/div/div[9]/div[2]/div[2]/div/div[2]/div[2]/div/div[1]/div[1]/div/div/div/div[1]/div[2]/ul[1]/li');

    links = []
    err_links = []

    for li in table:
        box = BeautifulSoup(li.get_attribute('innerHTML'), 'html.parser')
        #     print(box)
        box = box.find('div', {'class', 'resultHeader'})

        link = 'https://www-proquest-com.eproxy.lib.hku.hk/' + box.h3.find('a', href=True)['href']

        links.append(link)

    cnt = 1
    data = []

    for l in links:
        driver.execute_script('''window.open("about:blank", "_blank");''')
        driver.switch_to.window(driver.window_handles[cnt])

        driver.get(l)

        try:
            content = driver.find_element_by_xpath(
                '/html/body/div[4]/div[1]/div/div[9]/div[2]/div/div[1]/div[1]/article/div[4]/div/div/div/div/div[4]/div').text;

            driver.find_element_by_xpath(
                '/html/body/div[4]/div[1]/div/div[9]/div[2]/div/div[1]/div[1]/article/div[4]/ol/li[2]/a').click();

            time.sleep(3)
            table = driver.find_element_by_xpath(
                '/html/body/div[4]/div[1]/div/div[9]/div[2]/div/div[1]/div[1]/article/div[4]/div/div/div/div/div[2]/div[3]');

            table_contents = table.find_elements_by_xpath('div[@class="display_record_indexing_row"]');

            table_dict = {}

            for tc in table_contents:
                soup = BeautifulSoup(tc.get_attribute('innerHTML'), 'html.parser')
                table_dict[soup.find('div', {'class': 'display_record_indexing_fieldname'}).text.strip()] = soup.find(
                    'div',
                    {
                        'class': 'display_record_indexing_data'}).text

            table_contents_2 = table.find_elements_by_xpath('div[@class="display_record_indexing_row zebra-stripe"]');

            for tc in table_contents_2:
                soup = BeautifulSoup(tc.get_attribute('innerHTML'), 'html.parser')
                table_dict[soup.find('div', {'class': 'display_record_indexing_fieldname'}).text.strip()] = soup.find(
                    'div',
                    {
                        'class': 'display_record_indexing_data'}).text

            print(table_dict)

            data.append({
                'TITLE': table_dict['Title'],
                'UP_DT': datetime.now(),
                'PUB_DT': table_dict['Publication date'],
                'CONTENT': content,
                'DETAILS': str(table_dict)
            })

            time.sleep(0.01)

            driver.close()
            driver.switch_to.window(driver.window_handles[0])
        except NoSuchElementException:
            print('error')
            err_links.append(l)

    ProquestFt().insert_new_records_batch(data)
    driver.close()
