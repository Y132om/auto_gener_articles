#!/usr/bin/env python
# -*- coding:UTF-8 -*-

from keras.models import  Sequential
from keras.layers import Dense,LSTM,Embedding
from keras.callbacks import  LambdaCallback
import numpy as np
import random
import sys
import pickle


'''
数据预处理
1.读入数据，并设置编码
2.删除空行
'''
sentences=[]
with open('../data/小说.txt',encoding='utf8') as fr:
    lines=fr.readlines()
    for line in lines:
        line=line.strip()
        if line!=" ":
            sentences.append(line)

#文字映射
chars={}
for sentence in sentences:
    for c in sentence:
        chars[c]=chars.get(c,0)+1

chars=sorted(chars.items(),key=lambda x :x[1],reverse=True) #对值排序，从大到小
chars=[char[0] for char in chars]
vocab_size=len(chars)
print(vocab_size)

char2id={c:i for i,c in enumerate(chars)}
id2char={i:c for i,c in enumerate(chars)}

with open('novel.pkl','wb') as fw:
    pickle.dump([char2id,id2char],fw)



#训练数据
maxlen=10
step=3
embed_size=128
hidden_size=128
vocab_size=len(chars)
batch_size=64
epochs=20

X_data=[]
Y_data=[]

for sentence in sentences:
    for i in range(0,len(sentence)-maxlen,step):
        X_data.append([char2id[c] for c in sentence[i:i+maxlen]])
        y=np.zeros(vocab_size,dtype=np.bool)
        y[char2id[sentence[i+maxlen]]]=1
        Y_data.append(y)
X_data=np.array(X_data)
Y_data=np.array(Y_data)

print(X_data.shape,Y_data.shape)

model=Sequential()
model.add(Embedding(input_dim=vocab_size,output_dim=embed_size,input_length=maxlen))
model.add(LSTM(hidden_size,input_shape=(maxlen,embed_size)))
model.add(Dense(vocab_size,activation="softmax"))
model.compile(loss='categorical_crossentropy',optimizer='adam')


def sample(preds,diversity=1.0):
    preds=np.asarray(preds).astype('float64')
    preds=np.log(preds+1e-10)/diversity
    exp_preds=np.exp(preds)
    preds=exp_preds/np.sum(exp_preds)
    probas=np.random.multinomial(1,preds,1)
    return np.argmax(probas)

def on_epoch_end(epoch, logs):
    print('-' * 30)
    print('Epoch', epoch)

    index = random.randint(0, len(sentences))
    for diversity in [0.2, 0.5, 1.0]:
        print('----- diversity:', diversity)
        sentence = sentences[index][:maxlen]
        print('----- Generating with seed: ' + sentence)
        sys.stdout.write(sentence)

        for i in range(400):
            x_pred = np.zeros((1, maxlen))
            for t, char in enumerate(sentence):
                x_pred[0, t] = char2id[char]

            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = id2char[next_index]

            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()

model.fit(X_data,Y_data,batch_size=batch_size,epochs=epochs,callbacks=[LambdaCallback(on_epoch_end=on_epoch_end)])
model.save('novel_keras.h5')