import torch
import torch.autograd as autograd
import torch.nn.functional as F
import torch.nn as nn
import pdb

torch.manual_seed(1)

class DAN(nn.Module):
	def __init__(self, embeddings, args):
		super(DAN, self).__init__()

		self.args = args
		vocab_size, embed_dim = embeddings.shape

		self.embedding_layer = nn.Embedding( vocab_size, embed_dim)
		self.embedding_layer.weight.data = torch.from_numpy( embeddings )

		self.W_hidden = nn.Linear(embed_dim, args.hidden_size)
		self.W_out = nn.Linear(args.hidden_size, 2)

	def forward(self, x_indx):
		all_x = self.embedding_layer(x_indx)
		avg_x = torch.mean(all_x, dim=1)
		hidden = F.tanh(self.W_hidden(avg_x))
		out = F.log_softmax(self.W_out(hidden))
		return out
