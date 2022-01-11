import pandas as pd
import glob

mbti_type = ['enfp','estp', 'esfp', 'entp', 'estj', 'esfj', 'enfj', 'entj', 'istj', 'isfj', 'infj', 'intj', 'istp', 'isfp', 'infp', 'intp']

for i in mbti_type:
    each_type_paths = glob.glob('crawling/crawling_truity/*') #첨에 enfp 타입의 모든 크로링 데이터 가져옴
    df = pd.DataFrame()
    for each_type_path in each_type_paths: #만약 enfp하나에, 10개 데이터 있다 치면 10개 하나씩 가져와서
        df_temp = pd.read_csv(each_type_path, index_col=False)  #각각 csv로 변환 후
        df = pd.concat([df, df_temp]) #데이터프레임으로 콘캣


df.to_csv('./crawling_truity/all_mbti_Truity.csv', index=False) # mbti 하나 저장.

