import os
import sys
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
import torch.utils.data as data
from tqdm import tqdm
import datetime
import pdb
import numpy as np

def evaluate(output, target):
	_,guesses = output.data.topk(1)
	target = target.data
	return len(target) - (target-guesses.squeeze(1)).abs().sum()

def save_scores(scores, args):
	now = datetime.datetime.now()
	fp = open(str(now.hour)+str(now.minute)+'.res', 'wb')
	for attr, value in sorted(args.__dict__.items()):
		fp.write("{}={}\n".format(attr.upper(), value))
	fp.write('Epoch\tTrain\tDev\tTest\n')
	for i, (sc1, sc2, sc3) in enumerate(scores):
		fp.write('{}\t{}\t{}\t{}\n'.format(i,sc1,sc2,sc3))
	fp.close()

def train_model(train_data, dev_data, test_data, model, args):
	# use an optimized version of SGD 
	optimizer = torch.optim.Adam(model.parameters(),
								 lr=args.lr,
								 weight_decay=args.wd)

	#model.train()
	scores = []
	for epoch in range(1, args.epochs+1):
		print("-------------\nEpoch {}:\n".format(epoch))

		# train
		loss, guess, tot = run_epoch(train_data, True, model, optimizer, args)
		print('Train correct: {:.3f} ({}/{})'.format(float(guess)/tot,
													 guess,
													 tot))
		# dev
		dev_loss,dev_guess,dev_tot=\
								run_epoch(dev_data,False,model,optimizer,args)
		print('Dev correct: {:.3f} ({}/{})'.format(float(dev_guess)/dev_tot,
												   dev_guess,
												   dev_tot))

		# test
		test_loss,test_guess,test_tot=\
								run_epoch(test_data,False,model,optimizer,args)
		#print('Test MSE loss: {:.6f}'.format(test_loss))
		print('Test correct: {:.3f} ({}/{})'.format(float(test_guess)/test_tot,
													test_guess,
													test_tot))

		scores.append((float(guess)/tot,
					   float(dev_guess)/dev_tot,
					   float(test_guess)/test_tot))

	save_scores(scores,args)

def run_epoch(data, is_training, model, optimizer, args):
	# set loss function
	criterion = nn.NLLLoss()	

	# if not training, no need to use batches
	bs = len(data)
	if is_training: bs = args.batch_size
	
	# load random batches
	data_loader = torch.utils.data.DataLoader(
		data,
		batch_size=bs,
		shuffle=True,
		num_workers=4,
		drop_last=True)

	losses = []
	guesses = 0
	tot = 0

	# switch between training and evaluation modes; not needed here	
	#if is_training:
	#	model.train()
	#else:
	#	model.eval()

	# train on each batch
	for batch in tqdm(data_loader):

		# track inputs for auto-gradient calculation
		x, y = autograd.Variable(batch['x']), autograd.Variable(batch['y'])

		if is_training:
			# zero all gradients
			optimizer.zero_grad()

		# run the batch through the model
		out = model(x)
		# compute loss
		loss = criterion(out, y)
		# compute number correctly guessed
		guess = evaluate(out, y)

		if is_training:
			# back-propegate to compute gradient
			loss.backward()
			# descend along gradient
			optimizer.step()

		losses.append(loss.cpu().data[0])
		guesses+=guess
		tot += bs

	# Calculate epoch level scores
	avg_loss = np.mean(losses)
	return avg_loss, guesses, tot


