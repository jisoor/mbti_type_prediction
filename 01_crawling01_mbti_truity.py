from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException, StaleElementReferenceException
import pandas as pd
import re
import time

options = webdriver.ChromeOptions()
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')
options.add_argument('disable-gpu')
driver = webdriver.Chrome('./chromedriver', options=options)


next_button_xpath = '//*[@id="block-system-main"]/div/div[2]/a'
next_button_xpath_1 = '//*[@id="block-system-main"]/div/div/div[2]/a'
contents = []
type = []
mbti = ['enfp', 'estp', 'esfp', 'entp', 'estj', 'esfj', 'enfj', 'entj', 'istj', 'isfj', 'infj', 'intj', 'istp', 'isfp', 'infp', 'intp']
for i in mbti:
  url = 'https://www.truity.com/personality-type/{}'.format(i)
  driver.get(url)
  time.sleep(2)
  num = [1, 2, 4, 5, 7, 8, 10, 11]
  for j in num:
    try:
      overview = driver.find_element_by_xpath('//*[@id="block-system-main"]/div/div[1]/div/div[{}]/div'.format(j)).text
      contents.append(overview)
      type.append(i)
      print('append01')
    except:
      try:
        overview = driver.find_element_by_xpath('//*[@id="block-system-main"]/div/div[1]/div/div[{}]'.format(j)).text
        contents.append(overview)
        type.append(i)
      except:
        continue
  try:
    driver.find_element_by_xpath(next_button_xpath).click()
  except:
    driver.find_element_by_xpath(next_button_xpath_1).click()
  time.sleep(2)
  for k in range(1,4):
    try:
      strengths = driver.find_element_by_xpath('//*[@id="block-system-main"]/div/div[1]/div/div[{}]'.format(k)).text
      contents.append(strengths)
      type.append(i)
      print('append02')
    except:
      try:
        strengths = driver.find_element_by_xpath('//*[@id="block-system-main"]/div/div[1]/div/div[{}]/div'.format(k)).text
        contents.append(strengths)
        type.append(i)
      except:
        print('error02')
        continue

  try:
    driver.find_element_by_xpath(next_button_xpath).click()
  except:
    driver.find_element_by_xpath(next_button_xpath_1).click()
  time.sleep(2)
  L = [1,2,4,5,6]
  for l in L:
    try:
      career = driver.find_element_by_xpath('//*[@id="block-system-main"]/div/div[1]/div/div[{}]/div'.format(l)).text
      contents.append(career)
      type.append(i)
    except NoSuchElementException:
      continue
    except:
      break
  try:
    driver.find_element_by_xpath(next_button_xpath).click()
  except:
    driver.find_element_by_xpath(next_button_xpath_1).click()
  time.sleep(2)
  for m in range(1, 4):
    try:
      relationships = driver.find_element_by_xpath('//*[@id="block-system-main"]/div/div[2]/div/div[{}]/div'.format(m)).text
      contents.append(relationships)
      type.append(i)
    except NoSuchElementException:
      continue
    except:
      break


  df_mbti = pd.DataFrame(contents, columns=['comment'])
  df_mbti['type'] = type
  df_mbti.to_csv('./crawling_truity/{}_from_TRUITY'.format(i), index=False)
  contents = []
  type = []











