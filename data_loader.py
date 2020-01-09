import os
import torchvision.transforms as transforms
import torch.utils.data as data
import torch
import nltk
import pickle
import numpy as np
from PIL import Image
from build_vocab import Vocabulary
from pycocotools.coco import COCO



class CocoDataset(data.Dataset):
	"""COCO Custom Dataset compatible with torch.utils.data.DataLoader."""

	def __init__(self, root, json, vocab, transform=None):

		"""Set the path for images, captions and vocabulary wrapper."""
		self.root = root   #root is the image directory
		self.coco = COCO(json)
		self.ids = list(self.coco.anns.keys())
		self.vocab = vocab
		self.transform = transform

	def __getitem__(self, index):

		"""Returns one data pair (image and caption)."""
		coco = self.coco
		vocab = self.vocab
		ann_id = self.ids[index]
		caption = coco.anns[ann_id]['caption']
		img_id = coco.anns[ann_id]['image_id']
		path = coco.loadImgs(img_id)[0]['file_name']  #refer coco.loadImgs() 

		image = Image.open(os.path.join(self.root, path)).convert('RGB')  
		if self.transform is not None:
			image = self.transform(image)

		# Convert caption (string) to word ids
		tokens = nltk.tokenize.word_tokenize(str(caption).lower())
		caption = []
		caption.append(vocab('<start>'))
		caption.extend([vocab(token) for token in tokens])
		caption.append(vocab('<end>'))
		target = torch.Tensor(caption)
	  
		return image, target  #image is of tensor of shape(3,256,256) ; target is a tensor of shape (len(taget),) its a 1d array of indexes of each word in the target

	def __len__(self):
		return len(self.ids)


def collate_fn(data):
	"""modify the default collate funtion coz it doesnt add padding ; """
	
	"""    Args:
		data: list of tuple (image, caption). 
			- image: torch tensor of shape (3, 256, 256).
			- caption: torch tensor of shape (?); variable length.
	Returns:
		images: torch tensor of shape (batch_size, 3, 256, 256).
		targets: torch tensor of shape (batch_size, padded_length).
		lengths: list; valid length for each padded caption.
	# Sort a data list by caption length (descending order). """

	data.sort(key=lambda x: len(x[1]), reverse=True)
	images, captions = zip(*data)   # images is the list of all tensor(3d) images; captions is the list of all tensor(1d) type captions
	# Merge images (from tuple of 3D tensor to 4D tensor).
	images = torch.stack(images, dim=0)
	#merge captions (From 1d tensor to 2d tensor)
	lengths = [len(cap) for cap in captions]
	targets = torch.zeros(len(captions), max(lengths)).long()

	for i,cap in enumerate(captions):
		end = lengths[i]
		targets[i, :end] = cap[:end]
	return images, targets, lengths #the extra padded length has zeroes in it


def get_loader(root, json, vocab, transform, batch_size, shuffle, num_workers):
	
	coco = CocoDataset(root, json, vocab, transform)
	# This will return (images, captions, lengths) for each iteration.
	#images: a tensor of shape (batch_size, 3, 224, 224).
	# captions: a tensor of shape (batch_size, padded_length).
	# lengths: a list indicating valid length for each caption. length is (batch_size).
	data_loader = torch.utils.data.DataLoader(dataset=coco, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)
	return data_loader

