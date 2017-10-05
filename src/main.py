import argparse
import sys
import os
from os.path import dirname, realpath
# step back one directiory, to make data/ visible
sys.path.append(dirname(dirname(realpath(__file__))))
import data_utils
import model_utils
import train_utils
import torch
import pdb

# model parameters
parser = argparse.ArgumentParser(description=
								'PyTorch Example Sentiment Classifier')
parser.add_argument('--lr', type=float, default=0.001, help=
								'initial learning rate [default: 0.001]')
parser.add_argument('--epochs', type=int, default=50, help=
								'number of epochs for train [default: 256]')
parser.add_argument('--batch_size', type=int, default=173, help=
								'batch size for training [default: 173]')
parser.add_argument('--hidden_size', type=int, default=150, help=
								'size of hidden layer [default: 150]')
parser.add_argument('--wd', type=float, default=0, help=
								'weight decay parameter for Adam optimization')
args = parser.parse_args()

def main(args):
	print("\nParameters:")
	for attr, value in args.__dict__.items():
		print("\t{}={}".format(attr.upper(), value))

	# load data
	train_data, dev_data, test_data, embeddings = data_utils.load_dataset(args)

	# load model
	model = model_utils.DAN(embeddings, args)
	
	# train model
	train_utils.train_model(train_data, dev_data, test_data, model, args)

if __name__=="__main__":
	#for lr in [.00001, .001, .1, 10]:
	#	for hs in [75,150,300,600]:
	#		for wd in [.00001, .001, 10]:
	#			args.lr = lr
	#			args.hidden_size = hs
	#			args.wd = wd	
	main(args)
