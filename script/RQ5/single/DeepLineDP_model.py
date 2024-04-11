import torch
import numpy as np
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, Batch
from torch.utils.data import DataLoader, TensorDataset

import os
max_seq_len = 50
os.environ["TORCH_USE_CUDA_DSA"] = "1"


# Model structure
class HierarchicalAttentionNetwork(nn.Module):
    def __init__(self, vocab_size, embed_dim, word_gcn_hidden_dim, sent_gcn_hidden_dim, word_att_dim, sent_att_dim, use_layer_norm, dropout, device):
        super(HierarchicalAttentionNetwork, self).__init__()
        self.device = device
        self.sent_attention = SentenceAttention(
            vocab_size, embed_dim, word_gcn_hidden_dim, sent_gcn_hidden_dim, word_att_dim, sent_att_dim, use_layer_norm, dropout, device)

        self.fc = nn.Linear(sent_gcn_hidden_dim, 1)
        self.sig = nn.Sigmoid()

        self.use_layer_nome = use_layer_norm
        self.dropout = dropout

    def forward(self, code_tensor, word_edge, line_edge):

        code_lengths = []
        sent_lengths = []

        for file in code_tensor:
            code_line = []

            code_lengths.append(len(file))

            for line in file:
                code_line.append(len(line))

            sent_lengths.append(code_line)

        code_tensor = code_tensor.type(torch.LongTensor).to(self.device)
        code_lengths = torch.tensor(code_lengths).type(torch.LongTensor).to(self.device)
        
        code_embeds, word_att_weights, sent_att_weights, sents = self.sent_attention(code_tensor, code_lengths, word_edge, line_edge)

        scores = self.fc(code_embeds)
        final_scrs = self.sig(scores)

        return final_scrs, word_att_weights, sent_att_weights, sents

class SentenceAttention(nn.Module):
    """
    Sentence-level attention module. Contains a word-level attention module.
    """
    def __init__(self, vocab_size, embed_dim, word_gcn_hidden_dim, sent_gcn_hidden_dim, word_att_dim, sent_att_dim, use_layer_norm, dropout, device):
        super(SentenceAttention, self).__init__()
        self.device = device
        self.emb_dim = embed_dim
        self.dim = sent_gcn_hidden_dim

        self.word_attention = WordAttention(vocab_size, embed_dim, word_gcn_hidden_dim, word_att_dim, use_layer_norm, dropout, device)

        self.gcn = GCNConv(word_gcn_hidden_dim, sent_gcn_hidden_dim)

        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(sent_gcn_hidden_dim, elementwise_affine=True)
        self.dropout = nn.Dropout(dropout)

        # Sentence-level attention
        self.sent_attention = nn.Linear(sent_gcn_hidden_dim, sent_att_dim)

        # Sentence context vector u_s to take dot product with
        # This is equivalent to taking that dot product (Eq.10 in the paper),
        # as u_s is the linear layer's 1D parameter vector here
        self.sentence_context_vector = nn.Linear(sent_att_dim, 1, bias=False)

    def forward(self, code_tensor, code_lengths, word_edge, line_edge):

        packed_sents = code_tensor.reshape(sum(code_lengths), self.emb_dim)

        word_edge = [item for sublist in word_edge for item in sublist]

        # Word attention module
        sents, word_att_weights = self.word_attention(packed_sents, code_lengths, word_edge)  # [130, 50], [130]

        sents = self.dropout(sents)

        word_data_list = []
        for node, edge_index in zip(sents, word_edge):
            if np.any(edge_index == -1):
                edge_index = np.array([])
            edge_index = torch.tensor(edge_index, dtype=torch.long).to(self.device)
            word_data_list.append(Data(node, edge_index))   # node.shape [50, 50]  adj.shape [2, E]
        batch = Batch.from_data_list(word_data_list)

        line_out = self.gcn(batch.x, batch.edge_index)
        line_out = line_out.reshape((sum(code_lengths), max_seq_len, line_out.shape[1]))
        line_out = line_out.sum(dim=1)

        if self.use_layer_norm:
            normed_sents = self.layer_norm(line_out)
        else:
            normed_sents = line_out

        att = torch.tanh(self.sent_attention(normed_sents))
        att = self.sentence_context_vector(att).squeeze(1)
        val = att.max()
        att = torch.exp(att - val)
        att = att.reshape(len(code_lengths), int(len(att) / len(code_lengths)))
        sent_att_weights = att / torch.sum(att, dim=1, keepdim=True)

        code_tensor = line_out.reshape((len(code_lengths), int(len(line_out) / len(code_lengths)), line_out.shape[1]))

        code_tensor = code_tensor * sent_att_weights.unsqueeze(2)

        code_tensor = code_tensor.sum(dim=1)

        word_att_weights = word_att_weights.reshape((len(code_lengths), int(len(word_att_weights) / len(code_lengths)), word_att_weights.shape[1]))

        return code_tensor, word_att_weights, sent_att_weights, sents


class WordAttention(nn.Module):
    """
    Word-level attention module.
    """

    def __init__(self, vocab_size, embed_dim, gcn_hidden_dim, att_dim, use_layer_norm, dropout, device):
        super(WordAttention, self).__init__()
        self.device = device
        self.emb_dim = embed_dim
        self.embeddings = nn.Embedding(vocab_size, embed_dim)

        self.gcn = GCNConv(embed_dim, gcn_hidden_dim)

        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(gcn_hidden_dim, elementwise_affine=True)
        self.dropout = nn.Dropout(dropout)

        # Maps gru output to `att_dim` sized tensor
        self.attention = nn.Linear(gcn_hidden_dim, att_dim)

        # Word context vector (u_w) to take dot-product with
        self.context_vector = nn.Linear(att_dim, 1, bias=False)

    def init_embeddings(self, embeddings):
        """
        Initialized embedding layer with pretrained embeddings.
        embeddings: embeddings to init with
        """
        self.embeddings.weight = nn.Parameter(embeddings)

    def freeze_embeddings(self, freeze=False):
        """
        Set whether to freeze pretrained embeddings.
        """
        self.embeddings.weight.requires_grad = freeze

    def forward(self, sents, code_lenth, word_edge):
        """
        sents: encoded sentence-level data; LongTensor (num_sents, pad_len, embed_dim)
        return: sentence embeddings, attention weights of words
        """

        sents = self.embeddings(sents)
        # word_edge = [item for sublist in word_edge for item in sublist]

        word_data_list = []
        for node, edge_index in zip(sents, word_edge):
            if np.any(edge_index == -1):
                edge_index = np.array([])
            edge_index = torch.tensor(edge_index, dtype=torch.long).to(self.device)
            word_data_list.append(Data(node, edge_index))  # node.shape [50, 50]  adj.shape [2, E]
        batch = Batch.from_data_list(word_data_list)

        word_out = self.gcn(batch.x, batch.edge_index)  # [n*130, 50, 128]

        if self.use_layer_norm:
            normed_words = self.layer_norm(word_out)
        else:
            normed_words = word_out

        # Word Attenton
        att = torch.tanh(self.attention(normed_words))
        att = self.context_vector(att).squeeze(1)
        val = att.max()
        att = torch.exp(att - val)  # att.size: (n_words)  # [13000]
        att = att.reshape(sum(code_lenth), max_seq_len)
        att_weights = att / torch.sum(att, dim=1, keepdim=True)

        word_out = word_out.reshape((sum(code_lenth), max_seq_len, word_out.shape[1]))

        sents = word_out * att_weights.unsqueeze(2)


        return sents, att_weights