import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence

class EncoderCNN(nn.Module):
	def __init__(self,embed_size):
		super(EncoderCNN, self).__init__()
		resnet = models.resnet152(pretrained=True)
		modules = list(resnet.children())[:-1]
		self.resnet = nn.Sequential(*modules)
		self.linear = nn.Linear(resnet.fc.in_features, embed_size)
		self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

	def forward(self, images):
		with torch.no_grad():
			features = self.resnet(images)
		features = features.reshape(features.size(0), -1)
		features = self.linear(features)
		features = self.bn(features)   
		return features  #features = (batchsize, embedsize)


class DecoderRNN(nn.Module):
	def __init__(self,embed_size,hidden_size,vocab_size,num_layers,max_seq_length=20):
		super(DecoderRNN, self).__init__()
		self.embed = nn.Embedding(vocab_size, embed_size)
		self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
		self.linear = nn.Linear(hidden_size, vocab_size)
		self.max_seg_length = max_seq_length

	def forward(self, features, captions, lengths):
		#called during training
		embeddings = self.embed(captions)  #captions = (batchsize, paddedlength) ; embeddings = (batchsize, paddedlength, embedsize)
		embeddings = torch.cat((features.unsqueeze(1), embeddings), 1) #features.unsqueeze(1) = (batchsize, 1, embedsize) ; embeddings = (batchsize, paddedlength/seqlen, embedsize)
		packed = pack_padded_sequence(embeddings, lengths, batch_first=True) #shape ???????????????
		hiddens,_ = self.lstm(packed)  # hiddens = h_t(batchsize, paddedlength/seqlen(variable), embedsize)   ???????????????????????
		outputs = self.linear(hiddens[0])  #????????????????????
		return outputs

	def sample(self,features, states=None):
		#called during testing    
		sampled_ids = []
		inputs = features.unsqueeze(1)
		for i in range(self.max_seg_length):
			hiddens, states = self.lstm(inputs, states)
			hiddens = hiddens.squeeze(1)
			outputs = self.linear(hiddens)
			_, predicted = outputs.max(1)
			sampled_ids.append(predicted)
			inputs = self.embed(predicted)
			inputs = inputs.unsqueeze(1)
		sampled_ids = torch.stack(sampled_ids,1)
		return sampled_ids
	

			
		

	
