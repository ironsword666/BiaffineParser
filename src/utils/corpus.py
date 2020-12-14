# -*- coding: utf-8 -*-

from collections import Counter
from src.utils.vocab import Vocab

import torch



class Corpus(object):
    ''' Defines a general datatype.

    An example can be a sentence, a label sequence,  paired sentences ....
    '''

    def __init__(self):
        pass

    @classmethod
    def load(cls, path):
        return cls() 

    def save(self, path):
        pass

# TODO
class Sentence(object):

    def __init__(self):
        pass



class Conll(Corpus):
    '''

    A conllx file has 10 fields:
    1    ID      当前词在句子中的序号，１开始.
    2    FORM    当前词语或标点
    3    LEMMA   当前词语（或标点）的原型或词干，在中文中，此列与FORM相同
    4    CPOSTAG 当前词语的词性（粗粒度）
    5    POSTAG  当前词语的词性（细粒度）
    6    FEATS   句法特征，在本次评测中，此列未被使用，全部以下划线代替。
    7    HEAD    当前词语的中心词
    8    DEPREL  当前词语与中心词的依存关系
    '''

    FIELD_NAMES = ['ID', 'FORM', 'LEMMA', 'CPOSTAG', 'POSTAG', 'FEATS', 'HEAD', 'DEPREL', 'PHEAD', 'PDEPREL']

    def __init__(self, sentences):
        self.sentences = sentences

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx]

    def __getattr__(self, name):
        if not hasattr(self.sentences[0], name):
            raise AttributeError
        for s in self.sentences:
            yield getattr(s, name)
        
    # TODO __setattr__()

    @classmethod
    def load(cls, path):

        with open(path, 'r', encoding='UTF-8') as fr:
            lines = [line.strip() for line in fr]

        start, sentences = 0, []
        for i, line in enumerate(lines):
            if not line: 
                # [[id, form, ...], [id, form, ...], ...]
                sentence = [line.split('\t') for line in lines[start:i]]
                # [[1, 2, 3, ...], [In, an, Oct, ...], ...]
                values = [list(f) for f in zip(*sentence)]
                sentences.append(ConllSentence(Conll.FIELD_NAMES, values))
                start = i + 1
        
        return cls(sentences)

    def save(self, path):
        pass

class ConllSentence(Sentence):

    def __init__(self, field_names, values):
        for name, value in zip(field_names, values):
            setattr(self, name, value)
        self.field_names = field_names
        self.length = len(getattr(self, field_names[0]))

    def __len__(self):
        return self.length

    # TODO
    def __repr__(self):

        return None

# TODO
class Embedding(Corpus):

    def __init__(self, words, embeddings):

        self.vocab = Vocab(Counter(words))
        self.vectors = torch.tensor(embeddings)
        self.dim = self.vectors.size(-1)

    @classmethod
    def load(cls, path):

        words, embeddings = [], []
        # read pretrained embeddings
        with open(path, 'r', encoding='utf-8') as fr:
            for line in fr:
                row = line.strip().split(' ')  
                # the first line may contain the information of the file, ignore it  
                if len(row) < 2: 
                    continue
                words.append(row[0])
                embeddings.append([float(i) for i in row[1:]])
        
        return cls(words, embeddings)