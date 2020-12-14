from collections import defaultdict, Counter
from collections.abc import Iterable

from src.utils.common import unk_token

class Vocab(object):
    ''' Defines a vocabulary object that will be used to numericalize a field.
    
    Attributes:
        stoi: A collections.defaultdict instance mapping token strings to
            numerical identifiers.
        itos: A list of token strings indexed by their numerical identifiers.
    '''

    UNK = unk_token

    def __init__(self, counter=None, min_freq=1, 
                 specials=[]):

        self.itos = specials

        # frequencies of special tokens are not counted 
        for tok in specials:
            del counter[tok]
        
        self.itos.extend([tok for tok, freq in counter.items() 
                          if freq >= min_freq])
        
        # record how many tokens are in Vocab initially
        self.n_init = len(self.itos)
        
        if Vocab.UNK in specials:
            self.unk_index = specials.index(Vocab.UNK)
            self.stoi = defaultdict(self._default_unk_index)
        # TODO if Vocab.UNK not in specials
        else:
            # raise Exception('unk token doesn\'t match !')
            self.stoi =defaultdict()

        self.stoi.update({tok: idx for idx, tok in enumerate(self.itos)})

    def _default_unk_index(self):
        return self.unk_index

    def __getstate__(self):
        # avoid picking defaultdict
        attrs = dict(self.__dict__)
        # cast to regular dict
        attrs['stoi'] = dict(self.stoi)
        return attrs

    def __setstate__(self, state):
        stoi = defaultdict(self._default_unk_index)
        stoi.update(state['stoi'])
        state['stoi'] = stoi
        self.__dict__.update(state)

    def __getitem__(self, tok):
        return self.stoi[tok]

    def token2id(self, tokens):
        if not isinstance(tokens, Iterable):
            raise Exception('don\'t support one element mapping !')
        return [self.stoi[tok] for tok in tokens]

    def id2token(self, indices):
        if not isinstance(indices, indices):
            raise Exception('don\'t support one element mapping !')
        return [self.itos[idx] for idx in indices]

    def __len__(self):
        return len(self.itos)

    def extend(self, vocab):
        ''' extend current vocab by another vocab. '''

        for tok in vocab.itos:
            if tok not in self.stoi:
                self.itos.append(tok)
                self.stoi[tok] = len(self.itos) - 1

# l = ['a', 'b', 'c']
# c = Counter(l)
# for tok, freq in c.items():
#     print(freq)
# v = Vocab(c, specials=['<pad>', '<unk>'])
# print(v.token2id(['a','d']))






    




