import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import pickle
from tensorflow.keras.models import load_model
import nltk
nltk.download('omw-1.4')  #이코드들 없이 실행시, 오류가 뜨면서 다운로드 받으라고 친절히 코드까지 알려줌/
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
from nltk.tag import untag
from nltk.stem import WordNetLemmatizer
pd.set_option('display.unicode.''east_asian_width', True)

#X값은 따로 바꿔가며 주기
#Y값은 16개
X = '''Holland’s Peter Parker wants nothing more than to jump into the fray – a prominent feature of Extraversion. His social life is weak, but it’s not because he doesn’t want one. He may not be a social A-lister in high school, but it’s not because he prefers to be alone. It’s just that as a dedicated crime fighter, he has little time to cultivate many friendships outside of class. 
Peter’s eager to be accepted as a member of the Avengers, although he remains on the periphery. His banter is nonstop – not just while hiding behind a mask, like other versions of the character, but even as Peter Parker (see his back-and-forth with bodega owner Mr. Delmar in Homecoming.) He’s not only outgoing with people, but he’s also eager to enjoy all the experiences his unique powers offer him.'''
Y = ['enfp','estp', 'esfp', 'entp', 'estj', 'esfj', 'enfj', 'entj', 'istj', 'isfj', 'infj', 'intj', 'istp', 'isfp', 'infp', 'intp']

#target labeling
with open('../../Downloads/models/encoder.pickle', 'rb') as f: # 만들어놓은 encoder 불러오기
    encoder = pickle.load(f)
labeled_Y = encoder.transform(Y) # Y값을 encoder에 적용
print(encoder.classes_)
label = encoder.classes_
print(labeled_Y[:5])
print(label)

onehot_Y = to_categorical(labeled_Y) # 원핫인코딩 행렬화
print(onehot_Y)


# 형태소 분리, 한 글자/불용어 제거 ↓
# 1. tokenizing
tokened_X = nltk.word_tokenize(X)
print(tokened_X)

# 2. pos 태깅 ( word_tokenize 먼저 해야함)
possed_X = nltk.pos_tag(tokened_X)
print(possed_X)

# 3. lemmatzing  품사가 VB VBD VBG VBN VBP VBZ 애네만 골라서 동사 원형(lemmatizing)만들기
lm = WordNetLemmatizer()
for i in range(len(possed_X)):
    possed_X[i] = list(possed_X[i])
    if (possed_X[i][1] == "VB") | (possed_X[i][1] == "VBD") | (possed_X[i][1] == "VBG") | (possed_X[i][1] == "VBN") | (possed_X[i][1] == "VBP") | (possed_X[i][1] == "VBZ"):
        possed_X[i][0] = lm.lemmatize(possed_X[i][0], pos='v')

print(possed_X)

# 4. 모두 untag 해서 리스트 만들기
for i in range(len(possed_X)):
    possed_X[i] = possed_X[i][0]
print('untagged_lists: ', possed_X)
print(type(possed_X))

# 한글자인 단어 빼기, stopwords에 관형사 빼기
# 그 다음 tokenizing 시작

#  불용어, 한글자제거
stopwords = pd.read_csv('../../Downloads/stopwords(Eng).csv')
stopwords = stopwords.T  # stopwords단어에 하나하나 인덱싱 되어있어서 전치행렬 해줌(행-렬 교체교체)
stopwords.reset_index(inplace=True) #인덱스 제거
print(stopwords)
print(list(stopwords['index']))
for j in range(len(X)):
    words = []
    for i in range(len(X[j])):
        if len(X[j][i]) > 1:
            if X[j][i] not in list(stopwords['index']):
                words.append(X[j][i])
    X[j] = ' '.join(words)
words = []
for i in possed_X:
    if len(i) > 1:
        if i not in list(stopwords['index']):
            words.append(i)

X = ' '.join(words)
print('불용어 까지 제거한 X', X)

# titles tokenizing
with open('../../Downloads/models/mbti_token.pickle', 'rb') as f:
    token = pickle.load(f)
tokened_X = token.texts_to_sequences([X])
print(tokened_X)

if 1250 < len(tokened_X):
    tokened_X = tokened_X[:1250]

# padding
X_pad = pad_sequences(tokened_X, 1250)
print(X_pad)

# model.load
# model.predict(X_pad)
# predict과 onehot_Y와 비교

############### 여기까지함 ##########################

model = load_model('./models/mbti_classification_model_0.2888889014720917.h5')
preds = model.predict(X_pad)
print(label[np.argmax(preds)])

