import os
from collections import Counter

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.nn.utils.rnn import pad_sequence

class TextDataSet(Dataset):
    '''a TextDateset'''

    def __init__(self, args, sentences, word_vocab, char_vocab, rel_vocab):
        '''deal with the dataset'''

        super(TextDataSet, self).__init__()
        # from texts to numbers
        self.data = numericalize(args, sentences, word_vocab, char_vocab, rel_vocab)

    def __getitem__(self, index):
        '''return a piece of data based on the index'''

        return self.data[index]

    def __len__(self):
        '''the length of Dateset'''

        return len(self.data)

class TextDataLoader(DataLoader):
    '''加载TextDataset中的数据'''

    def __init__(self, batch_size, shuffle=False):
        pass


class TextBatchSampler(Sampler):
    '''批量采样数据'''

    def __init__(self, batch_size, shuffle=False):
        pass


def batchify(dataset, batch_size, shuffle=False):
    '''批量获得数据'''

    data_loader = DataLoader(dataset=dataset,
                        batch_size=batch_size,
                        shuffle=shuffle,
                        collate_fn=collate_fn)

    return data_loader

def collate_fn(data):
    '''define how dataLoader organize a batch data.
    
    Params:
        - data: [s1, s2, ...]
        - s1: (words, chars, heads, rels)
        - words: Tensor(sentence_len)
        - chars: Tensor(sentence_len, fix_len)
        - heads: Tensor(sentence_len)
        - rels:  Tensor(sentence_len)

    Returns:
        Here, words... are a batch of sentence

        words: Tensor(batch_size, max_len)
        chars: Tensor(batch_size, max_len, fix_len)
        heads: Tensor(batch_size, max_len)
        rels:  Tensor(batch_size, max_len)

    '''

    # sort the sentences
    data.sort(key=lambda s: len(s[0]), reverse=True)
    # split different part of a sentence: words, chars, heads, rels
    res = list(zip(*data)) 
    # TODO 
    # # get lens of sentences, we can also get it by mask
    # lens = [len(s) for s in words] 
    for i in range(len(res)):
        res[i] = pad_sequence(res[i], batch_first=True)

    # we can return the res directly, here just to be clear
    words, chars, heads, rels = res

    return words, chars, heads, rels


def numericalize(args, sentences, word_vocab, char_vocab, rel_vocab):
    '''transform a string sequence to a number sequnece.

    For example, (In, an, Oct., ...) -> [5, 108, 3999, ...].
    We should alse add 'bos' tag to begin of a sentence. 

    conllx format:
    1    ID      当前词在句子中的序号，１开始.
    2    FORM    当前词语或标点
    3    LEMMA   当前词语（或标点）的原型或词干，在中文中，此列与FORM相同
    4    CPOSTAG 当前词语的词性（粗粒度）
    5    POSTAG  当前词语的词性（细粒度）
    6    FEATS   句法特征，在本次评测中，此列未被使用，全部以下划线代替。
    7    HEAD    当前词语的中心词
    8    DEPREL  当前词语与中心词的依存关系
    '''

    data = []
    # to id
    word_to_id = {word:i for i, word in enumerate(word_vocab)}
    char_to_id = {char:i for i, char in enumerate(char_vocab)}
    rel_to_id = {rel:i for i, rel in enumerate(rel_vocab)}
    # add bos and numericalize
    for sentence in sentences:
        # words: Tensor(sentence_len)
        words = [word_to_id.get(word, args.unk_index) for word in [args.bos] + sentence[1]]
        words = torch.tensor(words)
        # chars: Tensor(sentence_len, fix_len)
        # to begin of the sentence, 'chars' of it is just <bos> of the char vocab
        chars = [[args.bos_index]] + [[char_to_id.get(char, args.unk_index) for char in word] for word in sentence[1]]
        # truncate if longer than fix_len, or pad if shorter than fix_len
        chars = [word[:args.fix_len] + [args.pad_index] * (args.fix_len - len(word)) for word in chars]
        # !! here, we treat a word as a whole, so don't seperate chars of a word 
        chars = torch.tensor(chars)
        # the heads of word s
        # TODO whether add bos ??
        heads = [int(head) for head in [args.bos_index] + sentence[6]]
        heads = torch.tensor(heads)
        # rels
        rels = [rel_to_id.get(rel, args.unk_index) for rel in [args.bos] + sentence[7]]
        rels = torch.tensor(rels)

        data.append((words, chars, heads, rels))

    return data
    








