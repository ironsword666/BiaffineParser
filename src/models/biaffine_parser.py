import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

from src.modules.mlp import MLP
from src.modules.biaffine import Biaffine


class BiaffineParser(nn.Module):
    '''
    Model(
    (word_embedding): Embedding(23144, 100)
    (bilstm): LSTM(200, 400, num_layers=3, batch_first=True, bidirectional=True)
    (mlp_arc_head): MLP(
        (linear): Linear(in_features=800, out_features=500, bias=True)
        (activation): LeakyReLU(negative_slope=0.1)
    )
    (mlp_arc_dep): MLP(
        (linear): Linear(in_features=800, out_features=500, bias=True)
        (activation): LeakyReLU(negative_slope=0.1)
    )
    (arc_attention): Biaffine()
    )
    '''

    def __init__(self, args):
        super(BiaffineParser, self).__init__()

        self.args = args
        self.pad_index = args.pad_index
        self.unk_index = args.unk_index
        
        # Embedding Layer
        self.word_embedding = nn.Embedding(num_embeddings=args.word_nums,
                                           embedding_dim=args.embedding_dim)

        # TODO charlstm, pretrained_embedding
        # self.char_lstm = 
        # if args.is_pre_trained:
        # self.pretrained_embedding = 

        # LSTM Layer
        self.bilstm = nn.LSTM(input_size=args.embedding_dim*2,
                              hidden_size=args.n_lstm_hidden,
                              num_layers=args.num_layers,
                              batch_first=True,
                              bidirectional=True)

        # MLP Layer
        # head 
        self.mlp_arc_head = MLP(n_in=args.n_lstm_hidden*2,
                             n_hidden=args.n_mlp_arc)

        # dependency
        self.mlp_arc_dep = MLP(n_in=args.n_lstm_hidden*2,
                             n_hidden=args.n_mlp_arc)

        # Biaffine Layer
        self.biaffine_arc = Biaffine(n_in=args.n_mlp_arc,
                                 n_out=1)


    def load_pretrained(self, embed=None):
        if embed is not None:
            self.pretrained_embedding = nn.Embedding.from_pretrained(embed)
            ## static or not?
            # self.pretrained_embedding.weight.requires_grad = False
        return self

    def forward(self, words, chars):
        # words not padded, mask: Tensor(batch, seq_len)
        mask = words.ne(self.pad_index)
        # actual length of sequence, lens:  Tensor(batch)
        lens = mask.sum(dim=1)

        # Embedding Layer
        # find words whose index is beyond word_embedding boundry
        inside_mask = words.ge(self.word_embedding.num_embeddings)
        # replace these indices with unk tag
        inside_words = words.masked_fill(inside_mask, self.unk_index)

        # (batch, seq_len) -> (batch, seq_len, embedding_dim)
        word_embed = self.word_embedding(inside_words)
        # pretrained embedding
        if hasattr(self, 'pretrained_embedding'):
            # (batch, seq_len, embedding_dim) + (batch, seq_len, embedding_dim)
            word_embed += self.pretrained_embedding(words)
        # TODO Char_LSTM 
        # TODO here just double word_embed
        # (batch, seq_len, embedding_dim*2)
        embed = torch.cat((word_embed, word_embed), dim=-1)

        # BiLSTM Layer
        x = pack_padded_sequence(embed, lens, batch_first=True)
        x, _ = self.bilstm(x)
        # (batch, seq_len, n_lstm_hidden*2)
        x, _ = pad_packed_sequence(x, batch_first=True)

        # MLP Layer
        # (batch, seq_len, n_lstm_hidden*2) -> (batch, seq_len, n_mlp_arc)
        arc_head = self.mlp_arc_head(x)
        arc_dep = self.mlp_arc_dep(x)

        # Biaffine Layer
        # ... -> (batch, seq_len, seq_len), with <bos> and <pad>
        # to a score matrix of size (seq_len, seq_len), s_ij is the score of j->i
        scores_arc = self.biaffine_arc(arc_dep, arc_head)


        # we should set scores that exceed the length of each sentence to -inf
        # (batch, seq_len, seq_len), with <bos> and <pad>
        scores_arc.masked_fill_(~mask.unsqueeze(1), float('-inf'))

        return scores_arc



      
        


