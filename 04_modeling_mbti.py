import pandas as pd
pd.options.display.max_rows = 20
pd.set_option('display.unicode.east_asian_width', True)
import warnings
warnings.filterwarnings(action='ignore')
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor='val_loss', patience=5)

X_train, X_test, Y_train, Y_test = np.load('../../Downloads/models/mbti_data_max_1250_wordsize_18714.npy', allow_pickle=True)
print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)

'''모델 생성'''
model = Sequential()
model.add(Embedding(18714, 300, input_length=1250))  # wordsize는 18714, max인 문장의 단어는 1250개 이므로 input_length=1250
model.add(Conv1D(256, kernel_size=5, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=1))
model.add(LSTM(128, activation='tanh', return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(64, activation='tanh', return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(64, activation='tanh'))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(16, activation='softmax'))  # 최종 y값이 16개 유형이므로 Dense(16), 다중분류 - softmax (binary는 sigmoid)
print(model.summary())

'''모델 학습'''
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) #다중분류는 categorical_crossentropy
fit_hist = model.fit(X_train, Y_train, batch_size=100,epochs=50, validation_data=(X_test,Y_test), callbacks=[early_stopping])

# 잊지말자 early stopping 넣는 법
# from keras.callbacks import EarlyStopping
# early_stopping = EarlyStopping(monitor='val_loss', patience=5)
# fit_hist = model.fit(X_train, Y_train, batch_size=100,epochs=50, validation_data=(X_test,Y_test), callbacks=[early_stopping])

'''모델 저장'''
model.save('./models/mbti_classification_model_{}.h5'.format(
    fit_hist.history['val_accuracy'][-1]))


print('\n\n=====DONE=====')
print('    code 0     ')
print('==============')