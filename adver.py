import tensorflow as tf
import numpy as np
import argparse
import matplotlib.pyplot as plt
import random

parser = argparse.ArgumentParser(description = 'Adversarial examples') # Creating a parser
parser.add_argument('-P','--predict', action='store_true', help='Predict image') # Optional Arguments for calling functions
parser.add_argument('-G','--fgsm', action='store_true', help='FGSM')
parser.add_argument('-F','--fgm', action='store_true', help='FGM')
parser.add_argument('-D','--deep_fool', action='store_true', help='Deep Fool')
parser.add_argument('-B','--benign_append', action='store_true', help='Benign Append')
parser.add_argument('-E','--evolution', action='store_true', help='Genetic algorithm')
parser.add_argument('-i','--image', type=str, metavar='', help='Image path') # Arguments for required variables
parser.add_argument('-e','--epsilon', type=str, metavar='', help='Epsilon/Eta')
parser.add_argument('-t','--target', type=str, metavar='', help='Target label (0-999)')
parser.add_argument('-m','--malware', type=str, metavar='', help='Malware file path')
parser.add_argument('-b','--benign', type=str, metavar='', help='Benign file path')
parser.add_argument('-c','--count', type=str, metavar='', help='max_iter/epoch')
parser.add_argument('-r','--mutation', type=str, metavar='', help='Mutation Rate')
parser.add_argument('-s','--size', type=str, metavar='', help='Population size')
args = parser.parse_args()

def preprocess(image):
	image = tf.cast(image, tf.float32)
	image = image/255
	image = tf.image.resize(image, (224, 224))
	image = image[None, ...]
	return image

def get_label(probs):
	return tf.keras.applications.mobilenet_v2.decode_predictions(probs, top=1)[0][0]

def display_image(image, image_probs):
	plt.figure()
	plt.imshow(image[0]*0.5+0.5) # To change [-1, 1] to [0,1]
	_, image_class, class_confidence = get_label(image_probs)
	plt.title('{} : {:.2f}% Confidence'.format(image_class, class_confidence*100))
	plt.show()

def predict_image(image_path):
	model = tf.keras.applications.MobileNetV2(include_top='True',weights='imagenet')
	model.trainable = False
	image_raw = tf.io.read_file(image_path)
	image = tf.image.decode_image(image_raw)
	image = preprocess(image)
	image_probs = model.predict(image)
	display_image(image, image_probs)

def FGSM(image_path, epsilon):
	model = tf.keras.applications.MobileNetV2(include_top='True',weights='imagenet')
	model.trainable = False
	image_raw = tf.io.read_file(image_path)
	image = tf.image.decode_image(image_raw)
	image = preprocess(image)
	y = model.predict(image)
	loss_object = tf.keras.losses.CategoricalCrossentropy()
	with tf.GradientTape() as tape:
		tape.watch(image)
		prediction = model(image)
		loss = loss_object(y, prediction)
	gradient = tape.gradient(loss, image)
	perturbation = tf.sign(gradient) # Get the sign of the gradients to create the perturbation
	adv_x = image + epsilon*perturbation
	adv_x = tf.clip_by_value(adv_x, -1, 1)
	display_image(adv_x, model.predict(adv_x))

def find_grad(model, image, target):
	loss_object = tf.keras.losses.CategoricalCrossentropy()
	with tf.GradientTape() as tape:
		tape.watch(image)
		prediction = model(image)
		loss = -loss_object(target, prediction)
	gradient = tape.gradient(loss, image)
	return gradient

def FGM(image_path, epsilon, max_iter, target):
	model = tf.keras.applications.MobileNetV2(include_top='True',weights='imagenet')
	model.trainable = False
	image_raw = tf.io.read_file(image_path)
	image = tf.image.decode_image(image_raw)
	image = preprocess(image)
	pert = tf.zeros_like(image)
	adv_x = image
	t = tf.keras.backend.one_hot(target, 1000)
	t = t[None, ...]
	loss_object = tf.keras.losses.CategoricalCrossentropy()
	for i in range(max_iter):
		delta = find_grad(model, adv_x, t)
		pert = pert + delta
		adv_x = image + pert*epsilon
		adv_x = tf.clip_by_value(adv_x, -1, 1)
	display_image(adv_x, model.predict(adv_x))

def Deep_Fool(image_path, eta, max_iter):
	model = tf.keras.applications.MobileNetV2(include_top='True',weights='imagenet')
	model.trainable = False
	image_raw = tf.io.read_file(image_path)
	image = tf.image.decode_image(image_raw)
	image = preprocess(image)
	y0 = tf.reshape(model(image), [-1])
	label = tf.argmax(y0)
	num_labels = y0.get_shape().as_list()[0]
	k_i = label
	count = 0
	pert = tf.zeros_like(image)
	adv_x = image
	score = np.inf
	while k_i == label and count < max_iter:
		with tf.GradientTape() as tape:
			tape.watch(adv_x)
			y = tf.reshape(model(adv_x), [-1])
			loss = y[label]
		org_grad = tape.gradient(loss, adv_x)
		for k in range(num_labels):
			if k == label:
				continue
			with tf.GradientTape() as tape:
				tape.watch(adv_x)
				y = tf.reshape(model(adv_x), [-1])
				loss = y[k]
			cur_grad = tape.gradient(loss, adv_x)
			w_k = cur_grad - org_grad
			y_k = tf.abs(y[k] - y[label])
			score_k = y_k/tf.norm(tf.reshape(w_k, [-1]))
			if score_k < score:
				score = score_k
				w = w_k
		dx = score*w
		pert = pert + dx
		adv_x = image + (1 + eta)*pert
		adv_x = tf.clip_by_value(adv_x, -1, 1)
		y = tf.reshape(model(adv_x), [-1])
		k_i = tf.argmax(y)
		count += 1
	display_image(adv_x, model.predict(adv_x))

def benign_append(malpath, benpath):
	model = tf.keras.models.load_model("../data/model/malconv.h5")
	with open(malpath, 'rb') as f:
		malcontent = f.read()
	maxlen = model.input_shape[1]
	X = np.ones((maxlen), dtype=np.uint16)*256
	byte = np.frombuffer(malcontent[:maxlen], dtype=np.uint8)
	X[:len(byte)] = byte
	X = np.asarray([X], dtype=np.uint16)
	print("[+] Original score is",model.predict(X)[0][0])
	if len(malcontent) < maxlen:
		with open(benpath, 'rb') as f:
			bencontent = f.read()
		ben = np.frombuffer(bencontent[:maxlen - len(malcontent)], dtype=np.uint8)
		X[0][len(malcontent) - maxlen:] = ben
		print("[+] After append the score is",model.predict(X)[0][0])
	else:
		print("[+] Cannot append, filesize exceeds max length")

##########----Trying to use Genetic algorithm to generate adversarial images-----############

def generate_members(image, min_val, max_val):
	pert = []
	for j in range(tf.size(image)):
		pert.append(random.uniform(min_val, max_val))
	return np.reshape(np.array(pert), tf.shape(image))

def crossover(a, b):
	temp1 = np.reshape(a, [-1])
	temp2 = np.reshape(b, [-1])
	index = random.randint(0, len(temp1) - 1)
	offspring1 = np.concatenate((temp1[:index], temp2[index:]))
	offspring2 = np.concatenate((temp2[:index], temp1[index:]))
	return np.reshape(offspring1, np.shape(a)), np.reshape(offspring2, np.shape(b))

def pairing(image, score, population, pop_size, min_val, max_val):
	score = np.array(score)
	temp = np.interp(score, (score.min(), score.max()), (0, 1))
	new_population = []
	count = 0
	difficulty = 1.75
	for i in range(len(score)):
		if random.random()*difficulty < temp[i]:
			for j in range(i+1, len(score)):
				if random.random()*difficulty < temp[j]:
					offspring1, offspring2 = crossover(population[i], population[j])
					new_population.append(offspring1)
					new_population.append(offspring2)
					count += 2
					if count >= pop_size - 1:
						return new_population
	return new_population

def mutation(offspring, mutation_rate, min_val, max_val):
	if random.random() < mutation_rate:
		temp = np.reshape(offspring, [-1])
		index = random.randint(0, len(temp)-1)
		temp[index] = random.uniform(min_val, max_val)
		temp = np.reshape(temp, np.shape(offspring))
		return temp
	else:
		return offspring

def Genetic_image(image_path, pop_size, mutation_rate, epsilon, target, epoch, min_val=-0.02, max_val=0.02):
	np.random.seed(1)
	random.seed(1)
	model = tf.keras.applications.MobileNetV2(include_top='True',weights='imagenet')
	model.trainable = False
	image_raw = tf.io.read_file(image_path)
	image = tf.image.decode_image(image_raw)
	image = preprocess(image)
	population = []
	for i in range(pop_size):
		population.append(generate_members(image, min_val, max_val))
	for j in range(epoch + 1):
		score = []
		for i in range(len(population)):
			adv_x = image + epsilon*population[i]
			adv_x = tf.clip_by_value(adv_x, -1, 1)
			y = model.predict(adv_x)
			score.append(y[0][target])
		if (j+1)%20 == 0:
			np.save("../data/model/gen"+str(j+1), population)
			print("[+] The maximum score after " + str(j+1) + " epochs is",max(score))
		if j == epoch:
			np.save("../data/model/genfin", population)
			break
		offsprings = pairing(image, score, population, pop_size, min_val, max_val)
		for i in range(len(offsprings)):
			offsprings[i] = mutation(offsprings[i], mutation_rate, min_val, max_val)
		offsprings.append(population[np.argmax(score)])
		while len(offsprings) < pop_size:
			offsprings.append(generate_members(image, min_val, max_val))
		population = offsprings
	pert = population[np.argmax(score)]
	adv_x = image + epsilon*pert
	display_image(adv_x, model.predict(adv_x)) 

if __name__ == '__main__':
	if args.predict:
		predict_image(args.image)
	elif args.fgsm:
		FGSM(args.image, float(args.epsilon))
	elif args.fgm:
		FGM(args.image, float(args.epsilon), int(args.count), int(args.target))
	elif args.deep_fool:
		Deep_Fool(args.image, float(args.epsilon), int(args.count))
	elif args.benign_append:
		benign_append(args.malware, args.benign)
	elif args.evolution:
		Genetic_image(args.image, int(args.size), float(args.mutation), float(args.epsilon), int(args.target), int(args.count))
	else:
		print("No argument. Use -h or --help for help.")