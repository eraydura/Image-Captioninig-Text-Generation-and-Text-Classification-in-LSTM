"""
! pip install -q kaggle
from google.colab import files
files.upload()
! mkdir ~/.kaggle
! cp kaggle.json ~/.kaggle/
! chmod 600 ~/.kaggle/kaggle.json
! kaggle datasets download -d adityajn105/flickr8k
! unzip flickr8k.zip -d train
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm.notebook import tqdm
from tensorflow.keras.applications.resnet import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding, add
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.models import Sequential
import datetime 
import cv2
import os
import tensorflow as tf
import logging
tf.get_logger().setLevel(logging.ERROR)
%matplotlib inline

def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
  
class Text_Generation():
  def __init__(self):
      self.text = open('text.txt', 'r').read()
      self.vocabulary = sorted(set(self.text))
      self.char2index = {c:i for i,c in enumerate(self.vocabulary)}
      self.int_text = np.array([self.char2index[i] for i in self.text])
      self.index2char = np.array(self.vocabulary)
      self.checkpoints= './training_checkpoints_LSTM'
      self.checkpoint = os.path.join(self.checkpoints, "checkpt_{epoch}") 
      self.log_dir = "logs/textgeneration/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
      self.seq_length= 150
      self.examples_per_epoch = len(self.text)
      self.char_dataset = tf.data.Dataset.from_tensor_slices(self.int_text)
      self.sequences = self.char_dataset.batch(self.seq_length+1, drop_remainder=True)
      self.dataset = self.sequences.map(self.create_input_target_pair)
      self.BATCH_SIZE = 128
      self.BUFFER_SIZE = 10000
      self.dataset = self.dataset.shuffle(self.BUFFER_SIZE).batch(self.BATCH_SIZE, drop_remainder=True)
      self.vocab_size = len(self.vocabulary)
      self.embedding_dim = 256
      self.rnn_units= 1024
      self.text_generated = []
      self.num_generate = 1000 
      self.EPOCHS=20

  def create_input_target_pair(self,chunk):
      input_text = chunk[:-1]
      target_text = chunk[1:]
      return input_text, target_text

  def build_model_lstm(self):
      model = Sequential()
      model.add(Embedding(self.vocab_size, self.embedding_dim,batch_input_shape=[self.BATCH_SIZE, None]))
      model.add(LSTM(self.rnn_units, return_sequences=True,stateful=True,recurrent_initializer='glorot_uniform'))
      model.add(Dense(self.vocab_size))
      return model

  def train(self):
    lstm_model = self.build_model_lstm()
    lstm_model.compile(optimizer=RMSprop(),metrics=['accuracy'], loss=loss)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=self.log_dir, histogram_freq=1)
    checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(filepath=self.checkpoint,save_weights_only=True)
    history = lstm_model.fit(self.dataset, epochs=self.EPOCHS, callbacks=[checkpoint_callback,tensorboard_callback,EarlyStopping(monitor='val_loss', patience=3)])
    self.generate_text()

  def generate_text(self):
      self.BATCH_SIZE = 1
      model = self.build_model_lstm()
      model.load_weights(tf.train.latest_checkpoint(self.checkpoints))
      model.build(tf.TensorShape([1, None]))
      start_string = input("Enter your starting string: ")
      input_eval = [self.char2index[s] for s in start_string]
      input_eval = tf.expand_dims(input_eval, 0)
      model.reset_states()
      for i in range(self.num_generate):
          predicted_id = tf.random.categorical(tf.squeeze(model(input_eval), 0) / 0.5, num_samples=1)[-1,0].numpy()
          input_eval = tf.expand_dims([predicted_id], 0)
          self.text_generated.append(self.index2char[predicted_id])
      print(start_string + ''.join(self.text_generated))

class Text_Clasification():
  def __init__(self):
        self.max_words = 1000
        self.max_len = 150
        self.BATCH_SIZE = 128
        self.embedding_dim = 256
        self.EPOCH=20
        self.log_dir = "logs/textclassification/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.df_train = pd.read_csv('train.txt', header =None, sep =';', names = ['input','sentiment'], encoding='utf-8')
        self.df_test = pd.read_csv('test.txt', header = None, sep =';', names = ['input','sentiment'],encoding='utf-8')
        self.hidden=[32,64,1024]
        self.data_plot()
        self.data_sequence()
    
  def data_plot(self):
        self.df_train.head()
        self.df_train.info()
        sns.countplot(self.df_train.sentiment)
        plt.xlabel('sentiment')
        plt.title('sentences')

  def data_sequence(self):
        self.X = self.df_train.input
        self.Y = self.df_train.sentiment
        le = LabelEncoder()
        self.Y = le.fit_transform(self.Y)
        self.Y = self.Y.reshape(-1,1)
        self.tok = Tokenizer(num_words=self.max_words)
        self.tok.fit_on_texts(self.X)
        sequences = self.tok.texts_to_sequences(self.X)
        self.sequences_matrix = pad_sequences(sequences,maxlen=self.max_len)

  def RNN(self,hidden_units):
        inputs = Input(name='inputs',shape=[self.max_len])
        layer = Embedding(self.max_words,self.embedding_dim,input_length=self.max_len)(inputs)
        layer = LSTM(hidden_units)(layer)
        layer = Dense(256,name='FC1')(layer)
        layer = Activation('relu')(layer)
        layer = Dropout(0.2)(layer)
        layer = Dense(6,name='out_layer')(layer)
        layer = Activation('sigmoid')(layer)
        model = Model(inputs=inputs,outputs=layer)
        model.compile(loss='mse', optimizer='adam')    
        return model    

  def train(self):
        results=[]
        for h in self.hidden:
          model = self.RNN(hidden_units=h)
          model.summary() 
          model.compile(loss='sparse_categorical_crossentropy',optimizer=RMSprop(),metrics=['accuracy'])
          tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=self.log_dir, histogram_freq=1)
          model.fit(self.sequences_matrix,self.Y,batch_size=self.BATCH_SIZE,epochs=self.EPOCH,
                    validation_split=0.2,callbacks=[tensorboard_callback,EarlyStopping(monitor='val_loss', patience=3)]) 
          self.test(model,results)

  def test(self,model,results):  
          X = self.df_test.input
          Y = self.df_test.sentiment
          le = LabelEncoder()
          Y = le.fit_transform(Y)
          Y = Y.reshape(-1,1)
          test_sequences = self.tok.texts_to_sequences(X)
          test_sequences_matrix = pad_sequences(test_sequences,maxlen=self.max_len)
          accr = model.evaluate(test_sequences_matrix,Y)
          results.append([accr[0],accr[1]])

class Image_Captioning():
  def __init__(self):
        self.features = {}
        self.mapping = {}
        self.all_cptns = []
        self.data()
        self.tokenizer = Tokenizer()
        self.tokenizer.fit_on_texts(self.all_cptns)
        self.vocab_size = len(self.tokenizer.word_index) + 1
        self.max_len = max(len(cptn.split()) for cptn in self.all_cptns)
        self.log_dir = "logs/imagecaptioning/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.batch_size=128
        self.img_ids = list(self.mapping.keys())
        self.train_data = self.img_ids[:int(len(self.img_ids) * 0.8)]
        self.test_data = self.img_ids[int(len(self.img_ids) * 0.8):]
        self.steps = len(self.train_data) // self.batch_size
        self.epoch=20
        self.img_files = os.listdir('/content/train/Images')

  def data(self):
        with open(os.path.join('/content/train/captions.txt'), 'r') as f:
            next(f)

            for line in f.read().split('\n'):
                tokens = line.split(',')
                img_id, cptns = tokens[0], tokens[1:]
                img_id = img_id.split('.')[0]
                cptns = " ".join(cptns)
                if img_id not in self.mapping:
                    self.mapping[img_id] = []
                self.mapping[img_id].append(cptns)

            for key, cptns in self.mapping.items():
                for i in range(len(cptns)):
                    caption = cptns[i].lower().replace('[^A-Za-z]', '').replace('\s+', ' ')
                    cptns[i] = 'startseq ' + " ".join([word for word in caption.split() if len(word)>1]) + ' endseq'

            for key in self.mapping:
                for cptn in self.mapping[key]:
                    self.all_cptns.append(cptn)

  def data_generator(self):
            x1, x2, y = [], [], []
            n = 0
            while True:
                for key in self.train_data:
                    n += 1
                    cptns = self.mapping[key]
                    for cptn in cptns:
                        seq = self.tokenizer.texts_to_sequences([cptn])[0]
                        for i in range(len(seq)):
                            in_seq, out_seq = seq[:i], seq[i]
                            in_seq = pad_sequences([in_seq], maxlen=self.max_len)[0]
                            out_seq = to_categorical([out_seq], num_classes=self.vocab_size)[0]
                            x1.append(self.features[key][0])
                            x2.append(in_seq)
                            y.append(out_seq)
                        
                    if n == self.batch_size:
                        x1, x2, y = np.array(x1), np.array(x2), np.array(y)
                        yield [x1,x2], y
                        x1, x2, y = [], [], []
                        n = 0 

  def training_model(self):
          resnet_model = ResNet50(include_top=True)
          resnet_model = Model(inputs=resnet_model.inputs, outputs=resnet_model.layers[-2].output)
          for img_name in tqdm(self.img_files):
              img = load_img('/content/train/Images/' + img_name, target_size=(224,224))
              feature = resnet_model.predict(preprocess_input(np.expand_dims(img_to_array(img), axis=0)), verbose=0)
              self.features[img_name.split('.')[0]] = feature
          input1 = Input(shape=(2048,))
          l1 = Dropout(0.1)(input1)
          l2 = Dense(1024, activation='relu')(l1)
          input2 = Input(shape=(self.max_len,))
          l3 = Embedding(self.vocab_size, 256, mask_zero=True)(input2)
          l4 = Dropout(0.1)(l3)
          l5 = LSTM(1024)(l4)
          dcdr1 = add([l2,l5])
          dcdr2 = Dense(1024, activation='relu')(dcdr1)
          output = Dense(self.vocab_size, activation = 'softmax')(dcdr2)
          model = Model(inputs=[input1, input2], outputs=output)
          self.train(model)

  def train(self,model):
          model.compile(loss='categorical_crossentropy', optimizer=RMSprop())
          generator = self.data_generator()
          tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=self.log_dir, histogram_freq=1)
          model.fit(generator, epochs=self.epoch, steps_per_epoch=self.steps, verbose=1, callbacks=[tensorboard_callback,EarlyStopping(monitor='val_loss', patience=3)])
          self.predict_captions(model)

  def word(self,intgr, tokenizer):
    for word, idx in tokenizer.word_index.items():
        if idx == intgr:
            return word
    return None

  def predict_captions(self, model):
        test_img_id = self.test_data[np.random.randint(0, len(self.test_data))]
        captions = self.mapping[test_img_id]
        in_text = 'startseq'
        for i in range(self.max_len):
            seq = self.tokenizer.texts_to_sequences([in_text])[0]
            seq = pad_sequences([seq], self.max_len)
            prd = model.predict([self.features[test_img_id], seq], verbose=0)
            prd = np.argmax(prd)
            word = self.word(prd, self.tokenizer)
            if word is None:
                break
            in_text += " " + word
            if word == 'endseq':
                break
        img = cv2.cvtColor(cv2.imread('/content/train/Images/' + test_img_id + '.jpg', 1), cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        print(in_text)

textgeneration=Text_Generation()
textgeneration.train()

textclass=Text_Clasification()
textclass.train()

image=Image_Captioning()
image.training_model()

  
