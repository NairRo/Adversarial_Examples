import tensorflow as tf
import numpy as np
import argparse
import matplotlib.pyplot as plt
import os
import argparse

parser = argparse.ArgumentParser(description = 'Append Attacks')
parser.add_argument('-F','--fgm',action='store_true',help='FGM Append Attack')
parser.add_argument('-G','--grad',action='store_true',help='Gradient Append Attack')
parser.add_argument('-B','--ben',action='store_true',help='Benign Append Attack')
parser.add_argument('-r','--rho',type=str,metavar='',help='value of rho for FGM attack')
parser.add_argument('-c','--count',type=str,metavar='',help='maximum iterations for gradient attack')
args = parser.parse_args()

def Mj(num,new_model):
	temp = np.ones((new_model.input_shape[1]), dtype=np.uint16)*256
	temp[0] = num
	temp = tf.convert_to_tensor([temp])
	return new_model.predict(temp)[0][0]

def grad_attack(embed_dict, rho, e, length, grad):
	eu = []
	for i in range(len(e)):
		eu.append(e[i] - rho*grad[i + length])
	return eu

def embedding_mapping(ex, embed_dict):
	x = []
	for i in range(len(ex)):
		min_dist = np.inf
		for j in range(256):
			dist = tf.math.reduce_euclidean_norm(ex[i] - embed_dict[j])
			if min_dist < dist:
				min_dist = dist
				temp = j
		x.append(j)
	return x


def Fgm(rho, mal, embed_dict, new, new1, model):
	with open('../data/Nov/'+mal, 'rb') as f:
		malcontent = f.read()
	maxlen = model.input_shape[1]
	X = np.ones((maxlen), dtype=np.uint16)*256
	byte = np.frombuffer(malcontent[:maxlen], dtype=np.uint8)
	X[:len(byte)] = byte
	X = np.asarray([X], dtype=np.uint16)
	length = len(malcontent)
	if length < maxlen:
		pad_length = maxlen - length
		if pad_length > 10000:
			pad_length = 10000
		for i in range(pad_length):
			X[0][i+length] = np.random.randint(0,256)
	X = tf.convert_to_tensor(X)

	original_predict = model.predict(X)
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
		if tf.norm(grad) == 0.0:
			return original_predict, original_predict
		ex = grad_attack(embed_dict, rho, e, length, grad)
		x = embedding_mapping(ex, embed_dict)
		X_ = tf.make_tensor_proto(X)
		X_ = tf.make_ndarray(X_)
		for i in range(pad_length):
			X_[0][length + i] = x[i]
		return original_predict,model.predict(X_)
	else:
		return original_predict, original_predict

def gradient_attack(count, mal, embed_dict, new, new1, model):
	with open('../data/Nov/'+mal, 'rb') as f:
		malcontent = f.read()
	maxlen = model.input_shape[1]
	X = np.ones((maxlen), dtype=np.uint16)*256
	byte = np.frombuffer(malcontent[:maxlen], dtype=np.uint8)
	X[:len(byte)] = byte
	X = np.asarray([X], dtype=np.uint16)
	length = len(malcontent)
	if length < maxlen:
		pad_length = maxlen - length
		if pad_length > 10000:
			pad_length = 10000
		for i in range(pad_length):
			X[0][i+length] = np.random.randint(0,256)
	X = tf.convert_to_tensor(X)

	original_predict = model.predict(X)
	if original_predict[0] < 0.5:
		return original_predict, original_predict
	elif length < maxlen:
		for k in range(count):
			y = new(X)
			with tf.GradientTape() as tape:
				tape.watch(y)
				pred = new1(y)
			grad = -tape.gradient(pred,y)[0]
			norm_grad = tf.norm(grad)
			X_ = tf.make_tensor_proto(X)
			X_ = tf.make_ndarray(X_)
			
			if norm_grad == 0.0:
				return original_predict,original_predict

			for i in range(pad_length):
				w = grad[i+length]
				n = w/norm_grad
				z = y[0][i + length]
				if tf.reduce_all(tf.equal(n,0)):
					continue
				d_min = np.inf
				for j in range(256):
					m = embed_dict[j]
					s = n*(m - z)
					d = tf.math.reduce_euclidean_norm(m - (z + s*n))
					if d < d_min:
						d_min = d
						new_byte = j
				X_[0][i+length] = new_byte
			X_ = tf.convert_to_tensor(X_)
			if model.predict(X_) < 0.5:
				break
			X = tf.make_tensor_proto(X_)
			X = tf.make_ndarray(X)
			X = tf.convert_to_tensor(X)
		return original_predict,model.predict(X_)
	else:
		return original_predict, original_predict

def benign_append(mal, ben, model):
	with open('../data/Nov/'+mal, 'rb') as f:
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
			with open('../data/benign/'+j, 'rb') as f:
				bencontent = f.read()
				ben = np.frombuffer(bencontent[:pad_length], dtype=np.uint8)
			for i in range(pad_length):
				X[0][i+length] = np.random.randint(0,256)
			temp = model.predict(X)[0]
			if temp < min_pred:
				min_pred = temp
		return original_predict,[min_pred]
	else:
		return original_predict,original_predict

def run_FGM_append(rho):
	model = tf.keras.models.load_model("../data/model/malconv.h5")
	#model.summary()
	idx = 2
	input_shape = model.layers[idx].input_shape
	layer_input = tf.keras.Input(shape=(input_shape[1],input_shape[2],))

	x = layer_input
	x1 = layer_input
	x = model.layers[2](x)
	x1 = model.layers[3](x1)
	z = model.layers[4]([x,x1])
	for layer in model.layers[5:]:
		z = layer(z)
	new1 = tf.keras.models.Model(layer_input, z)
	#new1.summary()

	layer_name = 'embedding_1'
	layer_output = model.get_layer(layer_name).output
	new = tf.keras.models.Model(model.input, outputs=layer_output)

	X_predicts = []
	X_new_predicts = []
	success = 0
	total_malware = 0
	total0 = 0
	total1 = 0
	malwares = os.listdir('../data/Nov')
	embed_dict = {}
	for i in range(257):
		embed_dict[i] = Mj(i,new)
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

def run_grad_append(count):
	model = tf.keras.models.load_model("../data/model/malconv.h5")
	idx = 2
	input_shape = model.layers[idx].input_shape
	layer_input = tf.keras.Input(shape=(input_shape[1],input_shape[2],))

	x = layer_input
	x1 = layer_input
	x = model.layers[2](x)
	x1 = model.layers[3](x1)
	z = model.layers[4]([x,x1])
	for layer in model.layers[5:]:
		z = layer(z)
	new1 = tf.keras.models.Model(layer_input, z)

	layer_name = 'embedding_1'
	layer_output = model.get_layer(layer_name).output
	new = tf.keras.models.Model(model.input, outputs=layer_output)

	X_predicts = []
	X_new_predicts = []
	success = 0
	total_malware = 0
	total0 = 0
	total1 = 0
	malwares = os.listdir('../data/Nov')
	embed_dict = {}
	for i in range(257):
		embed_dict[i] = Mj(i,new)
	for i in malwares:
		print('[+]',i)
		pred0,pred1 = gradient_attack(count,i,embed_dict,new,new1,model)
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

def run_benign():
	model = tf.keras.models.load_model("../data/model/malconv.h5")
	malwares = os.listdir('../data/Nov')
	ben = os.listdir('../data/benign')
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
		run_FGM_append(float(args.rho))
	elif args.grad:
		run_grad_append(int(args.count))
	elif args.ben:
		run_benign()
	else:
		print("Invalid arguments. Use -h or --help for help")
