import numpy as np
import pytesseract
from PIL import Image
from keras import backend as K

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow import keras
import tensorflow_hub as hub
from keras.models import load_model
import numpy as np
from testx import x


elmo_model = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)
batch_size = 61
max_len = 50
model = load_model('/content/drive/MyDrive/Colab-Model/elmo')


def ocr(path):
    text = pytesseract.image_to_string(Image.open(path))
    input_text = text.split('\n')
    input_text = [i.strip() for i in input_text]
    while('' in input_text):
        input_text.remove('')
    return input_text

def pred2label(pred):
    labels = ['company','total','address','date','O']
    out = []
    for pred_i in pred:
        out_i = []
        for p in pred_i:
            p_i = np.argmax(p)
            out_i.append(labels[p_i].replace("PADword","O"))
        out.append(out_i)
    return out


def pad(text):
    max_len = 50
    new_seq = []
    for i in range(max_len):
        try:
            new_seq.append(text[i])
        except:
            new_seq.append("PADword")
    return new_seq

def strip_text(input_text):
    input_text = input_text.split('\n')
    temp = []
    for i in input_text:
        temp.extend(list(i.split()))
    input_text = temp
    input_text = [i.strip() for i in input_text]
    while('' in input_text):
        input_text.remove('')
    return pad(input_text)

def compress(label,text):
    map_out = {}
    for i,t in zip(label,text):
        if i in map_out.keys():
            map_out[i].append(t)
        else:
            map_out[i] = [t]
    return map_out


def predict(img):
    text = pytesseract.image_to_string(Image.open(img))
    x[0] = text
    sess = tf.Session()
    K.set_session(sess)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())
    pred = model.predict(np.array(x),batch_size=batch_size)
    pred2label(pred[0])
    return compress(pred[0],text)
    