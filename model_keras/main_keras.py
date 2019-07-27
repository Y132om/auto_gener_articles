#!/usr/bin/env python
# -*- coding:UTF-8 -*-

from keras.models import load_model
import numpy as np
import pickle
import sys

maxlen=10
model=load_model('novel_keras.h5')

with open('novel.pkl','rb') as fr:
    [char2id,id2char]=pickle.load(fr)


def sample(preds,diversity=1.0):
    preds=np.asarray(preds).astype('float64')
    preds=np.log(preds+1e-10)/diversity
    exp_preds=np.exp(preds)
    preds=exp_preds/np.sum(exp_preds)
    probas=np.random.multinomial(1,preds,1)
    return np.argmax(probas)

sentence='把门推开，黑洞洞地，'
sentence=sentence[:maxlen]

diversity=1.0
print('----- Generating with seed: ' + sentence)
print('----- diversity:', diversity)
sys.stdout.write(sentence)

for i in range(400):
    x_pred=np.zeros((1,maxlen))
    for t,char in enumerate(sentence):
        x_pred[0,t]=char2id[char]

    preds=model.predict(x_pred,verbose=0)[0]
    next_index=sample(preds,diversity)
    next_char=id2char[next_index]

    sentence=sentence[1:]+next_char

    sys.stdout.write(next_char)
    sys.stdout.flush()