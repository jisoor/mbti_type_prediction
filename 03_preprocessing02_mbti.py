import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import pickle
from tensorflow.keras.utils import to_categorical
from konlpy.tag import Okt # 가 아니라 영문 nltk
import nltk


#### stopwords 영어버전 csv가 내장되어있는 모듈 불러와서 csv파일로 저장하기 ############
from nltk.corpus import stopwords   #corpus(말뭉치)라는 모듈에 stopwords가 내장되어있음.
import csv
nltk.download('stopwords')
stop_words = stopwords.words('english')
with open('../../Downloads/stopwords(Eng).csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(stop_words)
print(list(stop_words))


pd.set_option('display.unicode.east_asian_width', True)
df = pd.read_csv('../../Downloads/crawling/recompiled_all_mbti.csv')
print(df.head())
print(df.info())


X = df['comment']
Y = df['type']
#####  y값 전처리해서 encoder로 저장
encoder = LabelEncoder()
labeled_Y = encoder.fit_transform(Y) # enfp, estj 등의 타입을 숫자로 0~15까지로 만들어서 array타입에 저장
label = encoder.classes_
print(labeled_Y[:5])
print(label)  # encoder라벨정보가 리스트로 나옴
with open('../../Downloads/models/encoder.pickle', 'wb') as f:
    pickle.dump(encoder, f)
onehot_Y = to_categorical(labeled_Y) # 행렬로 만듬
print(onehot_Y)


# X값 형태소 분리 후 전처리

# nltk 영어 형태소 분리 파트 ↓↓↓↓
# ----------------------  tokenizing : 단어로 토크나이징 하기  ---------------------------
# 처음 사용하는 경우라면 먼저 nltk.download('punkt') 를 실행하여 Punkt Tokenizer Models (13MB) 를 다운로드 해줍니다.
# nltk.word_tokenize() 함수를 사용해서 괄호 안에 텍스트 문자열 객체를 넣어주면 Token으로 쪼개줍니다.
nltk.download('punkt')
for i in range(len(X)):
    X[i] = nltk.word_tokenize(X[i])
print('word_tokenize :', X)

# ------------------------ pos 태깅 (앞선 word_tokenize 이 선행되어야함) ------------------------------
nltk.download('averaged_perceptron_tagger')
for i in range(len(X)):
    X[i] = nltk.pos_tag(X[i])
print('pos 품사화 된 단어: ', X)

# ------------------------------------ lemmatzing 동사를 원형으로 만들기 ----------------------------------
# konlpy에서 stem=True로 어간을 제거했던것과 달리 영어는 문장마다 동사만 형태가 달라지므로 동사의 형태를 원형으로 복구해주는 것이 중요하다.
# 품사가 VB VBD VBG VBN VBP VBZ 애네가 모두 동사(verb)를 의미. 골라내서 동사 원형(lemmatizing)만들기
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

lm = WordNetLemmatizer() # 객체 생성
for i in range(len(X)):
    for j in range(len(X[i])):
        X[i][j] = list(X[i][j])  #튜플이니까 리스트로 바꿔야지!!
        if (X[i][j][1] == "VB") | (X[i][j][1] == "VBD") | (X[i][j][1] == "VBG")| (X[i][j][1] == "VBN")| (X[i][j][1] == "VBP")| (X[i][j][1] == "VBZ"):
            X[i][j][0] = lm.lemmatize(X[i][j][0], pos='v')
    print('동사가 전부 lemmatized된 pos: ', X[i])
print('X의타입', type(X))

# 모두 untag 해서 리스트 만들기
for i in range(len(X)):
    for j in range(len(X[i])):
        l = []
        X[i][j] = X[i][j][0]
        l.append(X[i][j])
    X[i] = l
print('튜플이 벗겨지고 리스트화된 단어들: ', X[0],X[1] )

# 한글자인 단어 빼기, stopwords(Eng)에 따로 불필요한 영단어 추가한 후 불용어 제거

stopwords = pd.read_csv('../../Downloads/stopwords(Eng).csv')
stopwords = stopwords.T
stopwords.reset_index(inplace=True)
print(stopwords.info())
print(stopwords.columns)
# exit()
for j in range(len(X)):
    words=[]
    for i in range(len(X[j])):
        if len(X[j][i]) > 1:
            if X[j][i] not in list(stopwords['index']):
                words.append(X[j][i])
    X[j] = ' '.join(words)
print('불용어 까지 제거한 X', X)

# 불용어 제거 잘되었나 한번 확인해보기
cleaned_df = pd.DataFrame(X, columns=['comment'])
cleaned_df.to_csv('./crawling/cleaned_nltk_mbti.csv', index=False)
print(cleaned_df.head())

# 토큰 생성

token = Tokenizer()
token.fit_on_texts(X)  # 고유한 단어에 각각 번호를 매겨 딕셔너리로 나타내줌
tokened_X = token.texts_to_sequences(X) # X값(comment값들)에다가 앞서 만든 딕셔너리 사전을 적용시켜줌
print(tokened_X[:5])

with open('../../Downloads/models/mbti_token.pickle', 'wb') as f: # token 저장
    pickle.dump(token, f)

# wordsize
wordsize = len(token.word_index) + 1
print(wordsize)
print(token.index_word)

# max 최대길이문장
max = 0
for i in range(len(tokened_X)):
    if max < len(tokened_X[i]):
        max = len(tokened_X[i])
print(max)

# pad_sequences로 문장에 패딩 입혀 행렬화 하기
X_pad = pad_sequences(tokened_X, max)
print(X_pad[:10])

# train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X_pad, onehot_Y, test_size=0.1)
print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)

# xy라는 이름으로 저장하기.
xy = X_train, X_test, Y_train, Y_test
np.save('./crawling/mbti_data_max_{}_wordsize_{}'.format(max, wordsize), xy)
