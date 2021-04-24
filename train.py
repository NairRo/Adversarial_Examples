import os
import numpy as np
import math
import random
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv1D, Activation, GlobalMaxPooling1D, Input, Embedding, Multiply
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras import metrics
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import LearningRateScheduler
from sklearn.model_selection import train_test_split

maxlen = 2**20
batch_size = 20
embedding_size = 8

def bytez_to_numpy(file):
	with open('../data/all_file/'+file,'rb') as f:
		content = f.read()
	X = np.ones((maxlen), dtype=np.uint16)*256
	byte = np.frombuffer(content[:maxlen],dtype=np.uint8)
	X[:len(byte)] = byte
	return X

def generator(data, labels, batch_size, shuffle=True):
	X = []
	Y = []
	zipped = list(zip(data, labels))
	while True:
		if shuffle:
			random.shuffle(zipped)
		for x,y in zipped:
			X.append(x)
			Y.append(y)
			if len(X) == batch_size:
				yield np.asarray(X,dtype=np.uint16), np.asarray(Y)
				X = []
				Y = []

inp = Input( shape=(maxlen,))
emb = Embedding( 257, embedding_size )( inp )
filt = Conv1D( filters=128, kernel_size=500, strides=500, use_bias=True, activation='relu', padding='valid' )(emb)
attn = Conv1D( filters=128, kernel_size=500, strides=500, use_bias=True, activation='sigmoid', padding='valid')(emb)
gated = Multiply()([filt,attn])
feat = GlobalMaxPooling1D()( gated )
dense = Dense(128, activation='relu')(feat)
outp = Dense(1, activation='sigmoid')(dense)

model = Model( inp, outp )
model.summary() 

model.compile( loss='binary_crossentropy', optimizer=SGD(lr=0.01,momentum=0.9,nesterov=True,decay=1e-3), metrics=[metrics.binary_accuracy] )

files = os.listdir('../data/all_file')
labels = []
file_bytes = []
for f in files:
	if 'benign' in f:
		labels.append(0)
		file_bytes.append(bytez_to_numpy(f))
	else:
		labels.append(1)
		file_bytes.append(bytez_to_numpy(f))

train_data, test_data, train_labels, test_labels = train_test_split(file_bytes, labels, test_size = 400)

train_gen = generator(train_data,train_labels,batch_size)
val_gen = generator(test_data,test_labels,batch_size)

base = K.get_value(model.optimizer.lr)
def schedule(epoch):
	return base/10.0**(epoch//2)

model.fit_generator(
	train_gen,
	steps_per_epoch = len(train_data)//batch_size,
	epochs = 10,
	validation_data = val_gen,
	callbacks = [LearningRateScheduler(schedule)],
	validation_steps = int(math.ceil(len(test_data)/batch_size)),
	)

print("success")

model.save("../data/model/malconv_final.h5")