import pandas as pd
import glob
# 각각 유형별로 콘캣 후 - > 16개 전부 콘캣
# 각각 유형에서 콘캣

# 1. personality database - 각각의 데이터 하나로 합치기 ex) enfp 유형에서 끌어온 모든 데이터 합치기
mbti_type = ['enfp','estp', 'esfp', 'entp', 'estj', 'esfj', 'enfj', 'entj', 'istj', 'isfj', 'infj', 'intj', 'istp', 'isfp', 'infp', 'intp']

for i in mbti_type:
     each_type_paths = glob.glob('./crawling/{}/*'.format(i)) #첨에 enfp 타입의 모든 크로링 데이터 가져옴
     df = pd.DataFrame()
     for each_type_path in each_type_paths: #만약 enfp하나에, 10개 데이터 있다 치면 10개 하나씩 가져와서
         df_temp = pd.read_csv(each_type_path, index_col=False)  #각각 csv로 변환 후
         df = pd.concat([df, df_temp]) #데이터프레임으로 콘캣

     df.dropna(inplace=True) # 난값 제거
     df.drop(['title'], inplace=True, axis=1)    # 'title'열 삭제
     df['type'] = '{}'.format(i)  # mbti 타입 열을 추가
     df.reset_index(drop=True, inplace=True) # 리셋 인덱스?
     df.to_csv('./crawling/final_crawling/all_mbti_{}.csv'.format(i), index=False) # mbti 하나 저장.

#  위에 거 다 하고 주석처리한 후 밑에 코드 실행
# 2. 16개의 mbti 유형 크롤링데이터 모두 콘캣

# data_paths = glob.glob('./crawling/final_crawling/*')
# df = pd.DataFrame()
# for data_path in data_paths:
#     df_temp = pd.read_csv(data_path, index_col=False)
#     df = pd.concat([df, df_temp])
# df.dropna(inplace=True)
# df.reset_index(drop=True, inplace=True)
# df.to_csv('./crawling/all_mbti_personality.csv', index=False)

# 위에 거 다 하고 주석처리한 후 밑에 코드 실행
# 3. 마지막 콘캣 : truity랑 personality database 사이트 두개 꺼 합치기
df_truity = pd.read_csv('./crawling/crawling_truity/all_mbti_Truity.csv')
df_personality = pd.read_csv('./crawling/all_mbti_personality.csv')
df = pd.concat([df_personality, df_truity])
df.to_csv('./crawling/final_all_mbti.csv', index=False)















