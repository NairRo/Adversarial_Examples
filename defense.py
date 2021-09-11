import numpy as np
import tensorflow as tf
import pefile
import os
import random

def raw_add(mal, maxlen, diction): 
	with open('../data/'+ diction +mal, 'rb') as f:
		malcontent = f.read()
	try:
		e = pefile.PE('../data/'+ diction +mal) #getting section information for binary file
		slack_indexes = []
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
		return slack_indexes
	except:
		print("fail",mal)
		return []

correct = 0
wrong = 0
true_positive = 0
false_positive = 0
failed = 0
total = 0
random.seed(1)

model = tf.keras.models.load_model("../data/model/malconv_final.h5") #loading the trained model
model.trainable = False
maxlen = model.input_shape[1]

files = os.listdir('../data/all_file')
random.shuffle(files)
ben = []
count = 0
for f in files:
	if 'benign' in f:
		ben.append(f)
		count += 1
	if count == 400: #randomly choosen 400 benigns
		break
advers = os.listdir('../data/adver_mals')

for i in ben:
	try:
		with open('../data/all_file/'+i, 'rb') as f:
			content = f.read()
		X = np.ones((maxlen), dtype=np.uint16)*256
		byte = np.frombuffer(content[:maxlen], dtype=np.uint8)
		X[:len(byte)] = byte
		X = np.asarray([X], dtype=np.uint16)
		pred = model.predict(X)
		if pred > 0.5:
			false_positive += 1
		else: # removing appended data
			pe = pefile.PE('../data/all_file/'+i)
			offset = pe.get_overlay_data_start_offset()
			with open('../data/all_file/'+i, 'rb') as f:
				content = f.read()
			content = content[:offset]
			X = np.ones((maxlen), dtype=np.uint16)*256
			byte = np.frombuffer(content[:maxlen], dtype=np.uint8)
			X[:len(byte)] = byte
			X = np.asarray([X], dtype=np.uint16)
			pred = model.predict(X)
			if pred > 0.5:
				false_positive += 1
			else: # changing slack bytes to  256
				byte = np.frombuffer(content[:maxlen], dtype=np.uint8)
				X = np.ones((maxlen), dtype=np.uint16)*256
				X[:len(byte)] = byte
				X = np.asarray([X], dtype=np.uint16)
				slack_indexes = raw_add(i,maxlen, "all_file/")
				for j in slack_indexes:
					X[0][j] = 256
				pred = model.predict(X)
				if pred > 0.5:
					false_positive += 1
				else:
					correct += 1
	except:
		correct += 1
		failed += 1
	total += 1

for i in advers:
	try:
		pe = pefile.PE('../data/adver_mals/'+i) # removing appended data (since they are adversarial examples we know they are detected as benign)
		offset = pe.get_overlay_data_start_offset()
		with open('../data/adver_mals/'+i, 'rb') as f:
			content = f.read()
		content_1 = content
		content = content[:offset]
		X = np.ones((maxlen), dtype=np.uint16)*256
		byte = np.frombuffer(content[:maxlen], dtype=np.uint8)
		X[:len(byte)] = byte
		X = np.asarray([X], dtype=np.uint16)
		pred = model.predict(X)
		if pred > 0.5:
			correct += 1
			true_positive += 1
		else: # changing slack bytes
			byte = np.frombuffer(content[:maxlen], dtype=np.uint8)
			X = np.ones((maxlen), dtype=np.uint16)*256
			X[:len(byte)] = byte
			X = np.asarray([X], dtype=np.uint16)
			slack_indexes = raw_add(i,maxlen, "adver_mals/")
			for j in slack_indexes:
				X[0][j] = 256
			pred = model.predict(X)
			if pred > 0.5:
				correct += 1
				true_positive += 1
	except:
		failed += 1
		with open('../data/adver_mals/'+i, 'rb') as f:
			content = f.read()
		X = np.ones((maxlen), dtype=np.uint16)*256
		byte = np.frombuffer(content[:maxlen], dtype=np.uint8)
		X[:len(byte)] = byte
		X = np.asarray([X], dtype=np.uint16)
		pred = model.predict(X)
		if pred > 0.5:
			correct += 1
			true_positive += 1
			print(i)
	total += 1

print("Accuracy:", correct/total*100)
print("Precision:", true_positive/(true_positive+false_positive)*100)
print("Recall:", true_positive/len(advers)*100)
print("Failed:",failed/len(advers)*100)
print(failed,total,len(advers))