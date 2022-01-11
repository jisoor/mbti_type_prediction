# 혼자 코드 분석한 파일,,,(무시해도됨)

############### read_csv ######################
test = pd.read_csv('./crawling/혼공.csv', sep='\t') # 구분자가 탭

test = pd.read_csv('./crawling/혼공.csv', index_col=0) #첫번째 열을 인덱스로 지정해주겠다. index_col = 'ID'도 가능, index_col=False 는 인덱스를 따로 지정하지 않음

############## dropna #########################
df.dropna(axis=0) #nan값이 있는 , 행 전체 제거
df.dropna(axis=1) # 열
df.dropna(inplace=True)

############ index 설정 및 리셋 #################
df.set_index('name', inplace=-True) # 'name"컬럼을 인덱스로 지정
df.reset_index(drop=True, inplace=True)  # 지정한 인덱스를 해제하고 인덱스는 걍 숫자가 됀, drop=True해주면 숫자 인덱스도 없어진ㅁ

############### Dataframe의 열 삭제하기  #################
df.drop(['title'], inplace=True, axis=1) # 열삭제, 행삭제는 axis=0 , axis='columns'라고 써줘도 됨

# 열 추가하기
df['title'] = 'enfp'  #이러면 모든 행이 'enfp' 라는 값을 가지는 열이 추가된다.