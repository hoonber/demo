import torch
from gensim.models import FastText
from torchcrf import CRF
import torch.nn as nn

class Attetion_model(nn.Module):
    def __init__(self, config):
        super(Attetion_model, self).__init__()
        drp = 0.1
        self.embedding_dim = config.embedding_dim
        self.hidden_dim = config.hidden_dim
        self.tagset_size = config.tagset_size
        self.use_fasttext = config.use_fasttext
        if self.use_fasttext:
            self.embed_model = config.embedding_model
        else:
            self.word_embeddings = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.num_layers = config.num_layers
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim, bidirectional=True, num_layers=self.num_layers)
        self.attention_layer = Attetion_model(10*2, self.maxlen)
        self.linear = nn.Linear(10* 2, self.tagset_size)
        self.hidden2tag = nn.Linear(2 * self.hidden_dim, self.tagset_size)
        self.crf = CRF(self.tagset_size)

    def forward(self, sentence, tags):
        if self.use_fasttext:
            embeds = self.get_embeddings(sentence)
        else:
            embeds = self.word_embeddings(sentence)

        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_space = self.hidden2tag(lstm_out)
        tag_scores = self.crf(tag_space, tags)
        return -1*(tag_scores) #return negative loglikelihood

    def get_lstm_features(self, sentence):
        if self.use_fasttext:
            embeds = self.get_embeddings(sentence)
        else:
            embeds = self.word_embeddings(sentence)

        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_sapce = self.hidden2tag(lstm_out)
        return tag_sapce

    def crf_decode(self, feats):
        return self.crf_decode(feats)

    def get_embedding(self, sentence):

        return torch.tensor(self.embed_model[sentence], dtype=torch.float)