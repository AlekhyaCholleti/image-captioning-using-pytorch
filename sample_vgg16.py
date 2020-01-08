import torch
import matplotlib.pyplot as plt
import numpy as np 
import argparse
import pickle 
import os
from torchvision import transforms 
from build_vocab import Vocabulary
from modelvgg import EncoderCNN, DecoderRNN
from PIL import Image


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_image(image_path, transform=None):
	image = Image.open(image_path)
	image = image.resize([224, 224], Image.LANCZOS)

	if transform is not None:
		image = transform(image)  #shape(3, 224, 224)   
		image = image.unsqueeze(0)   #shape(1, 3, 224, 224)

	return image
	
def main(args):
	transform = transforms.Compose([
		transforms.ToTensor(), 
		transforms.Normalize((0.485, 0.456, 0.406), 
							 (0.229, 0.224, 0.225))])

	#load vocabulary wrapper
	with open(args.vocab_path, 'rb') as f:
		vocab = pickle.load(f)   

	#build models
	encoder = EncoderCNN(args.embed_size).eval()
	decoder = DecoderRNN(args.embed_size, args.hidden_size, len(vocab), args.num_layers)	
	encoder = encoder.to(device)
	decoder = decoder.to(device)

	#load the trained model parameters
	encoder.load_state_dict(torch.load(args.encoder_path))
	decoder.load_state_dict(torch.load(args.decoder_path))

	#prepare an image
	image = load_image(args.image, transform)
	image_tensor = image.to(device)

	#generate a caption from image
	feature = encoder(image_tensor)
	sampled_ids = decoder.sample(feature)
	sampled_ids = sampled_ids[0].cpu().numpy()  
	#(1, max_seq_len) --> (max_Seq_length)

	#convert word_ids to words
	sampled_caption = []
	for word_id in sampled_ids:
		word = vocab.idx2word[word_id]
		sampled_caption.append(word)
		if word == '<end>':
			break
	sentence = ' '.join(sampled_caption)
	
	#print image and generated caption
	print(sentence)
	image = Image.open(args.image)
	plt.imshow(np.asarray(image))	

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--image', type=str, default='./png/women_dog2.jpg', help='input image for generating caption')
	parser.add_argument('--encoder_path', type=str, default='./data/models/encoder-3-3.ckpt', help='path for trained encoder')
	parser.add_argument('--decoder_path', type=str, default='./data/models/decoder-3-3.ckpt', help='path for trained decoder')
	parser.add_argument('--vocab_path', type=str, default='./data/vocab.pkl', help='path for vocabulary wrapper')
	
	# Model parameters (should be same as paramters in train.py)
	parser.add_argument('--embed_size', type=int , default=256, help='dimension of word embedding vectors')
	parser.add_argument('--hidden_size', type=int , default=512, help='dimension of lstm hidden states')
	parser.add_argument('--num_layers', type=int , default=1, help='number of layers in lstm')
	args = parser.parse_args()
	main(args)
