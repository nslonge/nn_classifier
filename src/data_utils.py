import gzip
import numpy as np
import torch
import torch.utils.data as data
import cPickle as pickle
import tqdm
import pdb

PATH="data/stsa.binary.{}"

# inherit from torch Dataset class
class FullDataset(data.Dataset):
	def __init__(self, name, word_to_indx, max_length=60):
		self.name = name
		self.dataset = []
		self.word_to_indx  = word_to_indx
		self.max_length = max_length

		with open(PATH.format(name)) as file:
			lines = file.readlines()
			for line in tqdm.tqdm(lines):
				sample = self.processLine(line)
				self.dataset.append(sample)
			file.close()

	## Convert one line from dataset to {Text, Tensor, Labels}
	def processLine(self, line):
		line = line.split()
		label = int(line[0])

		text = line[1:self.max_length+1]
		x =  getIndicesTensor(text, self.word_to_indx, self.max_length)
		sample = {'x':x, 'y':label}
		return sample

	def __len__(self):
		return len(self.dataset)

	def __getitem__(self,index):
		sample = self.dataset[index]
		return sample

# load each document into 1 x max_length tensor, with word index entries
def getIndicesTensor(text_arr, word_to_indx, max_length):
	nil_indx = 0
	text_indx = [word_to_indx[x] if x in word_to_indx 
								 else nil_indx for x in text_arr][:max_length]
	if len(text_indx) < max_length:
		text_indx.extend([nil_indx for _ in range(max_length-len(text_indx))])

	x =  torch.LongTensor(text_indx)

	return x

# load embedding for each word
def getEmbeddingTensor():
	embedding_path='data/word_vectors.txt.gz'
	lines = []
	with gzip.open(embedding_path) as file:
		lines = file.readlines()
		file.close()
	embedding_tensor = []
	word_to_indx = {}
	for indx, l in enumerate(lines):
		word, emb = l.split()[0], l.split()[1:]
		vector = [float(x) for x in emb ]
		# This is for the 'nil_index' words that aren't part
		# of the embedding dictionary
		if indx == 0:
			embedding_tensor.append( np.zeros( len(vector) ) )
		embedding_tensor.append(vector)
		word_to_indx[word] = indx+1
	embedding_tensor = np.array(embedding_tensor, dtype=np.float32)

	return embedding_tensor, word_to_indx

# Build dataset
def load_dataset(args):
	print("\nLoading data...")
	embeddings, word_to_indx = getEmbeddingTensor()
	args.embedding_dim = embeddings.shape[1]

	train_data = FullDataset('train', word_to_indx)
	dev_data = FullDataset('dev', word_to_indx)
	test_data = FullDataset('test', word_to_indx)
	return train_data, dev_data, test_data, embeddings

