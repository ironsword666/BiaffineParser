import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

class CharLSTM(nn.Module):
    '''use h_t of bilstm as representation of the word'''
    
    def __init__(self, n_chars, n_char_embed, n_out, pad_index=0):
        super(CharLSTM, self).__init__()
        
        self.pad_index = pad_index

        # Embedding Layer
        self.embedding = nn.Embedding(num_embeddings=n_chars,
                                      embedding_dim=n_char_embed)

        # LSTM Layer   
        self.lstm = nn.LSTM(input_size=n_char_embed,
                            hidden_size=n_out//2,
                            num_layers=1,
                            batch_first=True,
                            bidirectional=True)

    def forward(self, chars):
        '''
        Params:
            chars (Tensor(batch_size, seq_len, fix_len)):
        '''
        # Embedding Layer
        # mask: Tensor(batch_size, seq_len, fix_len)
        mask = chars.ne(self.pad_index)
        # lens: Tensor(batch_size, seq_len), word length, not sentence length
        lens = mask.sum(dim=-1)
        # exclude <pad> whose len is equal to 0
        char_mask = lens.gt(0)

        # chars[mask]: Tensor(batch_size, seq_len, fix_len) -> Tensor(n, fix_len), n = number of tokens except <pad> 
        # embed: Tensor(n, fix_len, n_char_embed)
        embed = self.embedding(chars[char_mask])

        # LSTM Layer
        # lens[char_mask]: Tensor(batch_size, seq_len) -> Tensor(n)
        x = pack_padded_sequence(embed, lens[char_mask], batch_first=True, enforce_sorted=False)
        # h: Tensor(2, n, n_out//2)
        x, (h, _) = self.lstm(x)
        # h: Tensor(n, n_out)
        h = torch.cat(h.unbind(), dim=-1)

        # transform Tensor(n, n_out) to Tensor(batch_size, seq_len, n_out)
        embed = h.new_zeros(*lens.size(), h.size()[-1])
        embed = embed.masked_scatter_(char_mask.unsqueeze(-1), h)

        return embed



                             