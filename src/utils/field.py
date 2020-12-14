from collections import Counter, OrderedDict

from src.utils.vocab import Vocab
from src.utils.dataloader import TextDataSet

import torch
from torch.nn.utils.rnn import pad_sequence


class RawField(object):
    ''' Defines a general datatype.

    Every dataset consists of one or more types of data. 
    For instance, a text classification dataset contains sentences and their classes, 
    while a machine translation dataset contains paired examples of text in two languages. 
    Each of these types of data is represented by a RawField object. 
    A RawField object does not assume any property of the data type and 
    it holds parameters relating to how a datatype should be processed.

    An example can be a sentence, a label sequence,  paired sentences ....
    '''

    def __init__(self, preprocessing=None, postprocessing=None):

        # self.name = name
        self.preprocessing = preprocessing
        self.postprocessing = postprocessing

    def preprocess(self, data):
        ''' Preprocess an example if `preprocessing` is provided. '''
        if self.preprocessing:
            return self.preprocessing(data)
        else:
            return data

    def process(self, batch):
        ''' Process a list of examples to create a batch. 
        
        Params:
            batch (List[object]): a list of examples.
        '''

        if self.postprocessing:
            batch = self.postprocessing(batch)
        return batch


class Field(RawField):
    '''
    
    Attributes:
        fix_len: A fixed length that all examples using this field will be padded to, or None for flexible sequence lengths.
        use_vocab: Whether to use a Vocab object. If False, the data in this field should already be numerical.
        stop_word: Tokens to discard during the preprocessing step.
        tokenize: The function used to tokenize a string into token sequences. Default: ``None``.
        lower: Whether to lowercase the text.
        preprocessing:  The function that will be applied to examples using this field after tokenizing but before numericalizing.
    '''

    def __init__(self, pad_token=None, unk_token=None, bos_token=None, eos_token=None, 
                 fix_len=None, use_vocab=True, stop_word=None, 
                 tokenize=None, lower=False, preprocessing=None, postprocessing=None):

        self.pad_token = pad_token
        self.unk_token = unk_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.fix_len = fix_len
        self.use_vocab = use_vocab
        self.stop_word = stop_word
        self.tokenize = tokenize
        self.lower = lower
        self.preprocessing = preprocessing
        self.postprocessing = postprocessing

    def preprocess(self, example):
        ''' Load a single example using this field, tokenizing if necessary.
        
        A example is like: 
            `Learning distributed representations of sentences from unlabelled data` which
            need to be split by blank chars;
            or chinese sentence `TorchText用法示例及完整代码` which should be segmented to words;
            even a word `postprocessing` which should be segmented to chars.
        '''

        if self.tokenize:
            example = self.tokenize(example)
        # TODO correct?
        if self.lower:
            example = [str.lower(w) for w in example]
        if self.use_vocab and self.stop_word is not None:
            example = [w for w in example if w not in self.stop_word]
        if self.preprocessing is not None:
            # such as: int() transform str to int; `booking` to `book`; ...
            example = self.preprocessing(example)

        return example


    def process(self, batch):
        ''' Process a list of examples to create a torch.Tensor. '''

        # TODO numericalize before dataloader or before dataset?
        tensors = self.numericalize(batch)
        padded = self.pad(tensors)
        
        return padded

    def build_vocab(self, examples, min_freq=1, specials=[], embed=None):
        ''' Construct the Vocab for this field from the dataset.

        Params:
            examples: Represent the set of possible values for this field.
            specials (list[str]): The list of special tokens.
        
        '''
        if not self.use_vocab:
            raise Exception('no need to build vocab !')

        # # TODO add `name` attribute to Field
        # if isinstance(dataset, TextDataSet):
        #     examples = [getattr(dataset, name) for name, field in 
        #                 dataset.fields.items() if field is self]
        # else:
        #     examples = dataset
        counter = Counter(token for example in examples for token in self.preprocess(example))

        # use OrderedDict to keep tokens ordered and unique
        specials = list(OrderedDict.fromkeys(
            token for token in [self.pad_token, self.unk_token, self.bos_token,
                                self.eos_token] + specials
            if token is not None))

        self.vocab = Vocab(counter, min_freq, specials)

        if embed is None:
            self.embed = None
        else:
            self.vocab.extend(embed.vocab)
            self.embed = torch.zeros(len(self.vocab), embed.dim)
            self.embed[self.vocab.token2id(embed.vocab.itos)] = embed.vectors
            # TODO z-norm


    def numericalize(self, batch):
        ''' numericalize a list of examples to create a torch.Tensor.

        Params: 
            batch (list[list(str)]): List of examples not tokenized and padded. 
        
        Returns:
            tensors (list[Tensor]): List of tensors, a tensor corresponding to a example numericalized. 
        '''

        batch = [self.preprocess(example) for example in batch]
        # print(batch[0])
        # TODO example is a tuple
        if self.bos_token:
            batch = [[self.bos_token] + example for example in batch]
        if self.eos_token:
            batch = [example + [self.eos_token] for example in batch]
        if self.use_vocab: 
            batch = [self.vocab.token2id(example) for example in batch]

        tensors = [torch.tensor(example) for example in batch]

        return tensors

    def pad(self, tensors):
        '''  
        Params:
            tensors (list[Tensor]): List of tensors to be padded.

        Returns:
            padded: (Tensor): tensors padded. 
        '''
        
        # TODO keep sorted
        return pad_sequence(tensors, batch_first=True).to(self.device)
    
    @property
    def device(self):
        return 'cuda' if torch.cuda.is_available() else 'cpu'

    # TODO check self.vocab exists
    @property
    def pad_index(self):
        return self.vocab[self.pad_token] if self.pad_token is not None else None

    @property
    def unk_index(self):
        return self.vocab[self.unk_token] if self.unk_token is not None else None

    @property
    def bos_index(self):
        return self.vocab[self.bos_token] if self.bos_token is not None else None
    
    @property
    def eos_index(self):
        return self.vocab[self.eos_token] if self.eos_token is not None else None

class SubWordField(Field):

    def __init__(self, **kwargs):
        super(SubWordField, self).__init__(**kwargs)

    def build_vocab(self, examples, min_freq=1, specials=[], embed=None):

        # TODO add `name` attribute to Field
        # if isinstance(dataset, TextDataSet):
        #     examples = [getattr(dataset, name) for name, field in 
        #                 dataset.fields.items() if field is self]
        # else:
        #     examples = dataset

        counter = Counter(char for example in examples 
                          for token in example 
                          for char in self.preprocess(token))

        # use OrderedDict to keep tokens ordered and unique
        specials = list(OrderedDict.fromkeys(
            token for token in [self.pad_token, self.unk_token, self.bos_token,
                                self.eos_token] + specials
            if token is not None))

        self.vocab = Vocab(counter, min_freq, specials)

        if embed is None:
            self.embed = None
        else:
            self.vocab.extend(embed.vocab)
            self.embed = torch.zeros(len(self.vocab), embed.dim)
            self.embed[self.vocab.token2id(embed.vocab.itos)] = embed.vectors


    def numericalize(self, batch):

        batch = [[self.preprocess(token) for token in example] for example in batch]
        # TODO deal with <bos> of sentence and <bos> of word, 
        # <bos> is in char vocab, then <bos> of sentence token2id as a number, 
        # not '< b o s >' split. 
        # And, there is no <bos> of word.
        # however, we can add <w> and </w> to improve performance.
        if self.bos_token:
            batch = [[[self.bos_token]] + example for example in batch]
        if self.eos_token:
            batch = [example + [[self.eos_token]] for example in batch]
        if self.use_vocab:
            batch = [[self.vocab.token2id(token) for token in example] for example in batch]
        batch = [[token[:self.fix_len] + [self.pad_index] * (self.fix_len - len(token)) 
                 for token in example] for example in batch]
        # list[Tensor(seq_len, fix_len)]
        batch = [torch.tensor(example) for example in batch]

        return batch