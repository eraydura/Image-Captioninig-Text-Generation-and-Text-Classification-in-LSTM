import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.models import Sequential
import datetime
import tensorflow as tf
# %matplotlib inline

class Text_Generation():
  def __init__(self,startString):
    self.text = open('text.txt', 'r').read() 
    self.vocabulary = sorted(set(self.text))
    self.vocab_size = len(self.vocabulary)
    self.index2char = np.array(self.vocabulary)
    self.embedding_dim = 256
    self.rnn_units= 1024
    self.BATCH_SIZE = 64
    self.BUFFER_SIZE = 10000
    self.EPOCH=50
    self.seq_length= 150
    self.startString=startString
    self.num_generate = 1000 
    self.data()

  def create_input_target_pair(self,chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text
        
  def data(self):
    self.char2index = {c:i for i,c in enumerate(self.vocabulary)}
    int_text = np.array([self.char2index[i] for i in self.text])
    char_dataset = tf.data.Dataset.from_tensor_slices(int_text)
    sequences = char_dataset.batch(self.seq_length+1, drop_remainder=True)
    dataset = sequences.map(self.create_input_target_pair)
    self.dataset = dataset.shuffle(self.BUFFER_SIZE).batch(self.BATCH_SIZE, drop_remainder=True)

  def model(self):
    model = Sequential()
    model.add(Embedding(self.vocab_size, 256,batch_input_shape=[64, None]))
    model.add(LSTM(1024,return_sequences=True,recurrent_initializer='glorot_uniform'))
    model.add(Dense(self.vocab_size, activation='softmax'))
    model.summary() 
    return model

  def train(self):
    model = self.model()
    model.summary()
    model.compile(optimizer=RMSprop(), loss='sparse_categorical_crossentropy')
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    history = model.fit(self.dataset, epochs=self.EPOCH,callback=[tensorboard_callback])
    self.test(model)

  def test(self,model):
        input_eval = tf.expand_dims([self.char2index[s] for s in self.startString], 0)
        text_generated = []
        model.reset_states()

        for i in range(self.num_generate):
            predictions = tf.squeeze(model(input_eval), 0) / 0.5
            predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()
            input_eval = tf.expand_dims([predicted_id], 0)
            text_generated.append(self.index2char[predicted_id])

        print(self.startString + ''.join(text_generated))

class Text_Clasification():
  def __init__(self):
        self.max_words = 1000
        self.max_len = 150
        self.BATCH_SIZE = 128
        self.embedding_dim = 256
        self.EPOCH=50
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
        layer = Dense(13,name='out_layer')(layer)
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
          log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
          tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
          model.fit(self.sequences_matrix,self.Y,batch_size=self.BATCH_SIZE,epochs=self.EPOCH,
                    validation_split=0.2,callbacks=[tensorboard_callback,EarlyStopping(monitor='val_loss',min_delta=1)]) 
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

textgeneration=Text_Generation()
textgeneration.train()

textclass=Text_Clasification()
textclass.train()
