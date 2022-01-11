from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException, StaleElementReferenceException
import pandas as pd
import re
import time
import pickle

# recompile... 영어만 남기고 싹 다 제거
pd.set_option('display.max_columns', 30)
df = pd.read_csv('../../Downloads/crawling/final_all_mbti.csv', index_col=False)
print(df.info())
print(df.columns)
print(type(df))
print(df.iloc[0, 1])


for i in range(6300):
    str_comment = df.iloc[i, 0]
    comment = re.compile('[^a-z|A-Z]').sub(' ', str_comment)
    df.iloc[i, 0] = comment


df.to_csv('./crawling/recompiled_all_mbti.csv', index=False)
print(df.head())
