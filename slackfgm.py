import pefile
import tensorflow as tf
import numpy as np
import os
import random
'''
with open('../data/all_file/mal_165', 'rb') as f:
	malcontent = f.read()
model = tf.keras.models.load_model("../data/model/malconv_final.h5") #loading the trained model
model.trainable = False
maxlen = model.input_shape[1]
X = np.ones((maxlen), dtype=np.uint16)*256
byte = np.frombuffer(malcontent[:maxlen], dtype=np.uint8)
length = len(malcontent)
X[:len(byte)] = byte
X = np.asarray([X], dtype=np.uint16)
original_predict = model.predict(X)
print(original_predict)
e = pefile.PE('../data/all_file/mal_165')
for section in e.sections:
	print(section.Name)
	print("\tVirtual Address: " + hex(section.VirtualAddress))
	print("\tVirtual Size: ",section.Misc_VirtualSize)
	print("\tRaw Size: ",section.SizeOfRawData)
	print("\tSize:",len(section.get_data()))

count = 0
for k in range(len(e.sections)):
	#print(e.sections[k].get_data()[(e.sections[k].Misc_VirtualSize-e.sections[k].SizeOfRawData):])
	if e.sections[k].SizeOfRawData > e.sections[k].Misc_VirtualSize:
		count += e.sections[k].SizeOfRawData - e.sections[k].Misc_VirtualSize
print(count)


for k in range(len(e.sections)):
	add = []
	if len(e.sections[k].get_data()) > 0: 
		for i in range(len(malcontent)-len(e.sections[k].get_data())+1):
			success = 1
			for j in range(len(e.sections[k].get_data())):
				if malcontent[i+j] != e.sections[k].get_data()[j]:
					success = 0
					break
			if success == 1:
				add.append(i)

		i = add[0]
		print("section",k,"add",i)
		if i != None:
			print("section",k,"size",len(e.sections[k].get_data()))
			for j in range(len(e.sections[k].get_data())):
				if malcontent[i+j] != e.sections[k].get_data()[j]:
					print("alert",i,j)
					break
'''

def raw_add(mal, maxlen): 
	with open('../data/all_file/'+mal, 'rb') as f:
		malcontent = f.read()
	try:
		e = pefile.PE('../data/all_file/'+mal) #getting section information for binary file
		slack_indexes = []
		slack_indexes_1 = []
		for k in range(len(e.sections)): # going through all the sections found in the binary file
			add = []
			if e.sections[k].SizeOfRawData > e.sections[k].Misc_VirtualSize and len(e.sections[k].get_data()) > (e.sections[k].SizeOfRawData - e.sections[k].Misc_VirtualSize): # if the raw size of the section is greater than virtual size and the difference is less than total data in the section
				none_zero = 0
				for i in e.sections[k].get_data(): # making sure the data has non zero element
					if i != 0: 
						none_zero = 1
						break
				if none_zero == 1:
					for i in range(len(malcontent)-len(e.sections[k].get_data())+1): # finding the location of the data in the binary file
						if (i + len(e.sections[k].get_data()) - (e.sections[k].SizeOfRawData- e.sections[k].Misc_VirtualSize)) >= maxlen:
							break
						success = 1
						for j in range(len(e.sections[k].get_data())):
							if malcontent[i+j] != e.sections[k].get_data()[j]:
								success = 0
								break
						if success == 1:
							add.append(i)
				if len(add) > 1: # if there are multiple locations for the data
					print("alert",k,mal,add)
				elif len(add) == 1: # if there is only one location
					for i in range(e.sections[k].SizeOfRawData - e.sections[k].Misc_VirtualSize):
						if (add[0]+len(e.sections[k].get_data())-i-1) < maxlen: # the index should be less than maximum size of input of model
							slack_indexes.append(add[0]+len(e.sections[k].get_data())-i-1)
							#slack_indexes.append(add[0]+e.sections[k].Misc_VirtualSize+i)
						#else:
						#	breaks

		return slack_indexes
	except:
		print("fail",mal)
		return []


def Mj(num,new_model): # function to get the embedded value of a byte
	temp = np.ones((new_model.input_shape[1]), dtype=np.uint16)*256
	temp[0] = num
	temp = tf.convert_to_tensor([temp])
	return new_model.predict(temp)[0][0]

def grad_attack(embed_dict, rho, e, signed_grad, slack_indexes):
	eu = []
	for i in range(len(e)):
		a = e[i] - rho*signed_grad[slack_indexes[i]]
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

def Slack(rho, mal, embed_dict, new, new1, model):
	with open('../data/all_file/'+mal, 'rb') as f: 
		malcontent = f.read()
	maxlen = model.input_shape[1]
	X = np.ones((maxlen), dtype=np.uint16)*256
	byte = np.frombuffer(malcontent[:maxlen], dtype=np.uint8)
	X[:len(byte)] = byte
	X = np.asarray([X], dtype=np.uint16)
	length = len(malcontent)

	original_predict = model.predict(X)
	print(original_predict)

	if original_predict[0] < 0.5:
		return original_predict, original_predict
	else:
		slack_indexes = raw_add(mal,maxlen) 
		#for i in slack_indexes:
		#	X[0][i] = np.random.randint(0,256) #found that appending random values in slack indexes at the start produces no difference in the result
		X = tf.convert_to_tensor(X)
		y = new(X)
		e = []
		for i in slack_indexes:
			e.append(embed_dict[X[0][i].numpy()]) # embedded value for bytes in slack indexes
		with tf.GradientTape() as tape:
			tape.watch(y)
			pred = new1(y)
		grad = tape.gradient(pred,y)[0] # gradient between output and embedded layer
		signed_grad = tf.sign(grad)
		if tf.norm(grad) == 0.0:
			return original_predict, original_predict
		ex = grad_attack(embed_dict, rho, e, signed_grad, slack_indexes)
		x = embedding_mapping(ex, embed_dict)
		X_ = tf.make_tensor_proto(X)
		X_ = tf.make_ndarray(X_)
		for i in range(len(slack_indexes)):
			X_[0][slack_indexes[i]] = x[i]
		print(original_predict,model.predict(X_))
		if model.predict(X_)< 0.5:
			X_ = X_.astype('uint8')
			f = open('../data/adver_mals_1/'+mal+'_slack', 'wb')
			f.write(X_[0][:length].tobytes())
		return original_predict,model.predict(X_)

def run_slack():
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
	malwares = []
	count = 0
	for f in files:
		if 'mal' in f:
			malwares.append(f)
	print(len(malwares))
	embed_dict = {}
	for i in range(257):
		embed_dict[i] = Mj(i,new) #dictionary that contains embedded mapping for all bytes
	for i in malwares:
		print('[+]',i)
		pred0,pred1 = Slack(1,i,embed_dict,new,new1,model)
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

run_slack()
