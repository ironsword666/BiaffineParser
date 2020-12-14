import os
from collections import Counter, OrderedDict

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.nn.utils.rnn import pad_sequence

class TextDataSet(Dataset):
    ''' TextDateset '''

    def __init__(self, examples, fields):
        ''' deal with the dataset.
        
        Params:
            examples (Corpus)
            fields (dict{Field: name}): name is corresponding name in Corpus
        '''

        super(TextDataSet, self).__init__()
        self.examples = examples
        self.fields = fields

    def __getitem__(self, idx):
        example = self.examples[idx]
        for name in self.fields.values():
            yield getattr(example, name)

    def __len__(self):
        return len(self.examples)

    def collect_fn(self, batch):
        ''' use f.process to numeriliaze and pad batch '''
        return {f: d for f, d in zip(self.fields.keys(), zip(*batch))}

    def build_loader(self, batch_size, shuffle=True):

        self.data_loader = TextDataLoader(dataset=self,
                                     batch_size=batch_size,
                                     shuffle=shuffle,
                                     collate_fn=self.collect_fn)


class TextDataLoader(DataLoader):
    '''加载TextDataset中的数据'''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.fields = self.dataset.fields

    def __iter__(self):
        # batch (dict{field:sub_batch}), result from TextDataSet.collect_fn 
        for batch in super().__iter__():
            # print(len(batch))
            yield [f.process(b) for f, b in batch.items()]
            # print('-------------')


class TextBatchSampler(Sampler):
    ''''''

    def __init__(self):
        pass


# def batchify(dataset, batch_size, shuffle=False):
#     '''批量获得数据'''

#     data_loader = DataLoader(dataset=dataset,
#                         batch_size=batch_size,
#                         shuffle=shuffle,
#                         collate_fn=collate_fn)

#     return data_loader

# def collate_fn(data):
#     '''define how dataLoader organize a batch data.
    
#     Params:
#         - data: [s1, s2, ...]
#         - s1: (words, chars, heads, rels)
#         - words: Tensor(sentence_len)
#         - chars: Tensor(sentence_len, fix_len)
#         - tags:  Tensor(sentence_len)

#     Returns:
#         Here, words... are a batch of sentence.

#         words: Tensor(batch_size, seq_len)
#         chars: Tensor(batch_size, seq_len, fix_len)
#         tags:  Tensor(batch_size, seq_len)

#     '''

#     # sort the sentences
#     data.sort(key=lambda s: len(s[0]), reverse=True)
#     # split different part of a sentence: words, chars, heads, rels
#     res = list(zip(*data)) 
#     # TODO 
#     # # get lens of sentences, we can also get it by mask
#     # lens = [len(s) for s in words] 
#     for i in range(len(res)):
#         res[i] = pad_sequence(res[i], batch_first=True)

#     # we can return the res directly, here just to be clear
#     words, chars, tags = res
    
#     return words, chars, tags











