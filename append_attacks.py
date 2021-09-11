import tensorflow as tf
import numpy as np
import argparse
import matplotlib.pyplot as plt
import os
import argparse
import random

parser = argparse.ArgumentParser(description = 'Append Attacks')
parser.add_argument('-F','--fgm',action='store_true',help='FGM Append Attack')
parser.add_argument('-G','--grad',action='store_true',help='Gradient Append Attack')
parser.add_argument('-B','--ben',action='store_true',help='Benign Append Attack')
parser.add_argument('-r','--rho',type=str,metavar='',help='value of rho for FGM attack')
parser.add_argument('-c','--count',type=str,metavar='',help='maximum iterations for gradient attack')
args = parser.parse_args()

def Mj(num,new_model): # function to get the embedded value of a byte
	temp = np.ones((new_model.input_shape[1]), dtype=np.uint16)*256
	temp[0] = num
	temp = tf.convert_to_tensor([temp])
	return new_model.predict(temp)[0][0]

def grad_attack(embed_dict, rho, e, length, signed_grad):
	eu = []
	for i in range(len(e)):
		a = e[i] - rho*signed_grad[i + length]
		eu.append(a)
	return eu

def embedding_mapping(ex, embed_dict):
	x = []
	for i in range(len(ex)):
		min_dist = np.inf
		for j in range(256):
			dist = tf.math.reduce_euclidean_norm(ex[i] - embed_dict[j])
			if min_dist > dist:
				min_dist = dist
				temp = j
		x.append(temp)
	return x


def Fgm(rho, mal, embed_dict, new, new1, model):
	with open('../data/all_file/'+mal, 'rb') as f:
		malcontent = f.read()
	maxlen = model.input_shape[1]
	X = np.ones((maxlen), dtype=np.uint16)*256
	byte = np.frombuffer(malcontent[:maxlen], dtype=np.uint8)
	X[:len(byte)] = byte
	X = np.asarray([X], dtype=np.uint16)
	length = len(malcontent)

	original_predict = model.predict(X)
	#print(original_predict)

	if length < maxlen:
		pad_length = maxlen - length
		if pad_length > 10000:
			pad_length = 10000
		for i in range(pad_length):
			X[0][i+length] = np.random.randint(0,256)
	X = tf.convert_to_tensor(X)

	if original_predict[0] < 0.5:
		return original_predict, original_predict
	elif length < maxlen:
		y = new(X)
		#print(new1.predict(y))
		#print(model.predict(X))

		e = []
		for i in range(pad_length):
			e.append(embed_dict[X[0][i+length].numpy()])
		with tf.GradientTape() as tape:
			tape.watch(y)
			pred = new1(y)
		grad = tape.gradient(pred,y)[0]
		signed_grad = tf.sign(grad)
		if tf.norm(grad) == 0.0:
			return original_predict, original_predict
		ex = grad_attack(embed_dict, rho, e, length, signed_grad)
		x = embedding_mapping(ex, embed_dict)
		X_ = tf.make_tensor_proto(X)
		X_ = tf.make_ndarray(X_)
		for i in range(pad_length):
			X_[0][length + i] = x[i]
		print(original_predict,model.predict(X_))
		if model.predict(X_)< 0.5:
			X_ = X_.astype('uint8')
			f = open('../data/adver_mals/'+mal+'_fgm', 'wb')
			f.write(X_[0][:length+pad_length].tobytes())
		return original_predict,model.predict(X_)
	else:
		return original_predict, original_predict

def gradient_attack(count, mal, embed_dict, new, new1, model):
	with open('../data/all_file/'+mal, 'rb') as f:
		malcontent = f.read()
	maxlen = model.input_shape[1]
	X = np.ones((maxlen), dtype=np.uint16)*256
	byte = np.frombuffer(malcontent[:maxlen], dtype=np.uint8)
	X[:len(byte)] = byte
	X = np.asarray([X], dtype=np.uint16)
	length = len(malcontent)

	original_predict = model.predict(X)
	#print(original_predict)

	if length < maxlen:
		pad_length = maxlen - length
		if pad_length > 10000:
			pad_length = 10000
		for i in range(pad_length):
			X[0][i+length] = np.random.randint(0,256)
	X = tf.convert_to_tensor(X)

	if original_predict[0] < 0.5:
		return original_predict, original_predict
	elif length < maxlen:
		for k in range(count):
			print(k,model.predict(X))
			y = new(X)
			with tf.GradientTape() as tape:
				tape.watch(y)
				pred = new1(y)
			grad = -tape.gradient(pred,y)[0]
			X_ = tf.make_tensor_proto(X)
			X_ = tf.make_ndarray(X_)

			for i in range(pad_length):
				w = grad[i+length]
				n = w/tf.math.reduce_euclidean_norm(w)
				if tf.reduce_all(tf.equal(n,0)):
					continue
				z = y[0][i + length]
				d_min = np.inf
				new_byte = X[0][i+length]
				for j in range(256):
					if j == X[0][i+length]:
						continue
					m = embed_dict[j]
					s = n*(m - z)
					d = tf.math.reduce_euclidean_norm(m - (z + s*n))
					if d < d_min and tf.reduce_all(tf.greater_equal(s,0)):
						d_min = d
						new_byte = j
				X_[0][i+length] = new_byte
			X_ = tf.convert_to_tensor(X_)
			pred_X_ = model.predict(X_)
			if pred_X_ < 0.5 or pred_X_ == model.predict(X):
				if model.predict(X_)< 0.5:
					X_ = tf.make_tensor_proto(X_)
					X_ = tf.make_ndarray(X_)
					X_ = X_.astype('uint8')
					f = open('../data/adver_mals/'+mal+'_grad', 'wb')
					f.write(X_[0][:length+pad_length].tobytes())
				break
			X = tf.make_tensor_proto(X_)
			X = tf.make_ndarray(X)
			X = tf.convert_to_tensor(X)
		print(original_predict,pred_X_)
		return original_predict,model.predict(X_)
	else:
		print("length and maxlen",length,maxlen)
		return original_predict, original_predict

def benign_append(mal, ben, model):
	with open('../data/all_file/'+mal, 'rb') as f:
			malcontent = f.read()
	maxlen = model.input_shape[1]
	X = np.ones((maxlen), dtype=np.uint16)*256
	byte = np.frombuffer(malcontent[:maxlen], dtype=np.uint8)
	length = len(malcontent)
	X[:len(byte)] = byte
	X = np.asarray([X], dtype=np.uint16)

	original_predict = model.predict(X)
	if original_predict[0] < 0.5:
		return original_predict, original_predict
	elif length < maxlen:
		pad_length = maxlen - length
		if pad_length> 10000:
			pad_length = 10000
		for j in ben:
			min_pred = 1
			with open('../data/all_file/'+j, 'rb') as f:
				bencontent = f.read()
				benign = np.frombuffer(bencontent[:pad_length], dtype=np.uint8)
			for i in range(pad_length):
				X[0][i+length] = benign[i]
			temp = model.predict(X)[0]
			if temp < min_pred:
				min_pred = temp
		return original_predict,[min_pred]
	else:
		return original_predict,original_predict

def run_FGM_append(rho):
	model = tf.keras.models.load_model("../data/model/malconv_final.h5") #loading the trained model
	#model.summary()
	model.trainable = False
	idx = 2 
	input_shape = model.layers[idx].input_shape #getting input shape of embedding layer
	layer_input = tf.keras.Input(shape=(input_shape[1],input_shape[2],))

	x = layer_input
	x1 = layer_input
	x = model.layers[2](x)
	x1 = model.layers[3](x1)
	z = model.layers[4]([x,x1])
	for layer in model.layers[5:]:
		z = layer(z)
	new1 = tf.keras.models.Model(layer_input, z) #second half of model(from embedding layer to output)
	#new1.summary()

	layer_name = 'embedding'
	layer_output = model.get_layer(layer_name).output
	new = tf.keras.models.Model(model.input, outputs=layer_output) #first half(from input to embedding layer)

	X_predicts = []
	X_new_predicts = []
	success = 0
	total_malware = 0
	total0 = 0
	total1 = 0
	files = os.listdir('../data/all_file')
	random.shuffle(files)
	malwares = []
	count = 0
	for f in files:
		if 'mal' in f:
			malwares.append(f)
			count += 1
		if count > 399: #randomly choosen 400 malwares
			break
	embed_dict = {}
	for i in range(257):
		embed_dict[i] = Mj(i,new) #dictionary that contains embedded mapping for all bytes
	for i in malwares:
		print('[+]',i)
		pred0,pred1 = Fgm(rho,i,embed_dict,new,new1,model)
		X_predicts.append(pred0[0])
		X_new_predicts.append(pred1[0])
		if pred0[0] > 0.5:
			total0 += 1
			total_malware += 1
			if pred1[0] < 0.5:
				success += 1
		if pred1[0] > 0.5:
			total1 += 1
	print("Percentage of successful adversarial examples:",success/total_malware*100)
	print("Original Percentage of malware detected:",total0/len(X_predicts)*100)
	print("Afer attack the percentage of malware detected:",total1/len(X_predicts)*100)

def run_grad_append(counter):
	model = tf.keras.models.load_model("../data/model/malconv_final.h5") #loading the trained model
	#model.summary()
	model.trainable = False
	idx = 2
	input_shape = model.layers[idx].input_shape #getting input shape of embedding layer
	layer_input = tf.keras.Input(shape=(input_shape[1],input_shape[2],))

	x = layer_input
	x1 = layer_input
	x = model.layers[2](x)
	x1 = model.layers[3](x1)
	z = model.layers[4]([x,x1])
	for layer in model.layers[5:]:
		z = layer(z)
	new1 = tf.keras.models.Model(layer_input, z) #second half of model(from embedding layer to output)
	#new1.summary()

	layer_name = 'embedding'
	layer_output = model.get_layer(layer_name).output
	new = tf.keras.models.Model(model.input, outputs=layer_output) #first half(from input to embedding layer)

	X_predicts = []
	X_new_predicts = []
	success = 0
	total_malware = 0
	total0 = 0
	total1 = 0
	files = os.listdir('../data/all_file')
	random.shuffle(files)
	malwares = []
	count = 0
	for f in files:
		if 'mal' in f:
			malwares.append(f)
			count += 1
		if count > 49: #randomly choosen 50 malwares
			break
	embed_dict = {}
	for i in range(257):
		embed_dict[i] = Mj(i,new) #dictionary that contains embedded mapping for all bytes
	o = 1
	for i in malwares:
		print('[+]',o,i)
		pred0,pred1 = gradient_attack(counter,i,embed_dict,new,new1,model)
		X_predicts.append(pred0[0])
		X_new_predicts.append(pred1[0])
		if pred0[0] > 0.5:
			total0 += 1
			total_malware += 1
			if pred1[0] < 0.5:
				success += 1
		if pred1[0] > 0.5:
			total1 += 1
		o += 1
	print("Percentage of successful adversarial examples:",success/total_malware*100)
	print("Original Percentage of malware detected:",total0/len(X_predicts)*100)
	print("Afer attack the percentage of malware detected:",total1/len(X_predicts)*100)

def run_benign():
	model = tf.keras.models.load_model("../data/model/malconv.h5")
	files = os.listdir('../data/all_file')
	random.shuffle(files)
	malwares = []
	count = 0
	for f in files:
		if 'mal' in f:
			malwares.append(f)
			count += 1
		if count > 399: #randomly choosen 400 malwares
			break
	ben = []
	count = 0
	for f in files:
		if 'benign' in f:
			ben.append(f)
			count += 1
		if count > 99: #randomly choosen 100 malwares
			break
	X_predicts = []
	X_new_predicts = []
	success = 0
	total_malware = 0
	total0 = 0
	total1 = 0
	for i in malwares:
		print('[+]',i)
		pred0,pred1 = benign_append(i,ben,model)
		X_predicts.append(pred0[0])
		X_new_predicts.append(pred1[0])
		if pred0[0] > 0.5:
			total0 += 1
			total_malware += 1
			if pred1[0] < 0.5:
				success += 1
		if pred1[0] > 0.5:
			total1 += 1
	print("Percentage of successful adversarial examples:",success/total_malware*100)
	print("Original Percentage of malware detected:",total0/len(X_predicts)*100)
	print("Afer attack the percentage of malware detected:",total1/len(X_predicts)*100)


if __name__ == '__main__':
	if args.fgm:
		for i in range(5):
			random.seed(i)
			run_FGM_append(float(args.rho))
	elif args.grad:
		for i in range(5):
			random.seed(i)
			run_grad_append(int(args.count))
	elif args.ben:
		for i in range(5):
			random.seed(i)
			run_benign()
	else:
		print("Invalid arguments. Use -h or --help for help")
