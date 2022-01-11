from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException, StaleElementReferenceException
import pandas as pd
import re
import time

# all mbti type 한꺼번에 크롤링
options = webdriver.ChromeOptions()
# options.add_argument('lang=ko_KR') # 영어라 필요 없는 코드
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')
options.add_argument('disable-gpu')
driver = webdriver.Chrome('./chromedriver', options=options)

# 어떠카지? 한 페이지에 총 몇개의 프로필이 있는지 모름(계속 밑으로 스크롤링 되고 숫자가 안나옴) -> 무한스크롤 코드
titles = []
comments = []
next_button_xpath = '//*[@id="root"]/div/section/main/div[1]/div[2]/div[1]/div/div[1]/div[5]/div[1]/span[1]'
review_number_xpath = '//*[@id="root"]/div/section/main/div[1]/div[2]/div[1]/div/div[1]/div[5]/h2' # 리뷰버튼 클릭
mbti_type = ['enfp', 'estp', 'esfp', 'entp', 'estj', 'esfj', 'enfj', 'entj', 'istj', 'isfj', 'infj', 'intj', 'istp', 'isfp', 'infp', 'intp'] # 16개 유형
# enfp로 검색한 페이지부터 크롤링 시작.

for m in mbti_type:
    url = 'https://www.personality-database.com/search?keyword={}'.format(m)
    driver.get(url)
    time.sleep(2)
    #  #리뷰갯수 형태는  'Typology Comments 5437'
    for l in range(1):  #페이지를 1페이지만 가져올 것임 (원래는 최소 1000페이지 이상인데 데이터가 과다)
        for i in range(1+(30*l),1+(30*(l+1))): # enfp로 검색한 페이지의 총 프로필수가 몇개인지 모르므로 1000으로 설정. 30개씩
            try:
                time.sleep(1)
                title_xpath ='//*[@id="root"]/div/section/main/div[1]/div[2]/div[1]/div/div[1]/div[3]/a[{}]/div/div[2]/h2'.format(i)
                title = driver.find_element_by_xpath(title_xpath).text
                print(title)

                driver.find_element_by_xpath(title_xpath).click()
                time.sleep(12)
                driver.find_element_by_xpath(next_button_xpath).click()
                time.sleep(4)
                review_number = driver.find_element_by_xpath(review_number_xpath).text
                review_number = int(review_number[18:])  # 'Typology Comments 5437' 이런형태로 쓰여있는 글자에서 뒤에 숫자만 따온 후 갯수만큼 크롤링 할것이다,
                print(review_number)
                for k in range(review_number // 10 + 1):
                    for j in range(k*10+1,  (k+1)*10+1):  # 리뷰의 갯수, 10개씩
                        try:
                            comment_xpath = '//*[@id="root"]/div/section/main/div[1]/div[2]/div[1]/div/div[1]/div[5]/div[2]/div[{}]/div/div[2]/div[1]/div[1]'.format(j)
                            comment = driver.find_element_by_xpath(comment_xpath).text
                            comments.append(comment)
                            titles.append(title)
                        except NoSuchElementException:
                            comment_xpath_1 = '//*[@id="root"]/div/section/main/div[1]/div[2]/div[1]/div/div[1]/div[5]/div[2]/div[{}]/div/div[2]/div[1]/div'.format(j)
                            try:
                                comment_1 = driver.find_element_by_xpath(comment_xpath_1).text
                                comments.append(comment_1)
                                titles.append(title)
                            except:
                                print('not exist')
                        except:
                            continue
                    # 스크롤 내리는 코드  import time
                    # Get scroll height
                    try:
                        while True:
                            last_height = driver.execute_script("return document.body.scrollHeight")        #[1]
                            # Scroll down to bottom                                                     #[2]
                            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                            # Wait to load page
                            time.sleep(3)                                                                #[3]
                            driver.execute_script("window.scrollTo(0, document.body.scrollHeight-50);")     #[4]
                            time.sleep(3)
                            # Calculate new scroll height and compare with last scroll height          #[5]
                            new_height = driver.execute_script("return document.body.scrollHeight")
                            if new_height == last_height:                                              #[6]
                                break
                            else:
                                last_height = new_height
                    except:
                        print('scroll_error')
                        break
        # [ 1 ] : 마지막 시점의 창 높이 저장
        # [ 2 ] : 창 높이까지 스크롤
        # [ 3 ] : 스크롤 후 창이 로딩될때까지 2초를 기다리겠다는 명령어. 로딩이 다되면 바로 넘어감
        # [ 4 ] : 한 번에 맨 마지막까지 스크롤되면 아래 리스트가 뜨지 않아서, 마지막을 찍고 조금 창을 올리는 방법으로 리스트가 로딩될 수 있게 함
        # [ 5 ] : 스크롤이 된 후의 창 높이를 새로운 높이로 저장
        # [ 6 ] : 새로운 높이가 이전 높이와 변하지 않았으면 스크롤 종료
                driver.get(url)
                print('get_url')
                time.sleep(2)
                df_enfp = pd.DataFrame(titles, columns=['title'])
                df_enfp['comment'] = comments
                df_enfp.to_csv('./crawling/{}/{}_{}번째프로필.csv'.format(m, m, i), index=False)
                titles = []
                comments = []
            except:
                print('empty title')
                continue
        # 스크롤 내리는 코드
        try:
            last_height = driver.execute_script("return document.body.scrollHeight")  # [1]
            # Scroll down to bottom                                                     #[2]
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            # Wait to load page
            time.sleep(2)  # [3]
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight-50);")  # [4]
            time.sleep(2)
            # Calculate new scroll height and compare with last scroll height          #[5]
            new_height = driver.execute_script("return document.body.scrollHeight")

            if new_height == last_height:  # [6]
                break
            else:
                last_height = new_height
        except:
            break
        driver.get(url)


    #스크롤 내리는 코드
 # 스크롤 내리는 코드  j는 10번 i는 30번



print(len(titles))
driver.close()