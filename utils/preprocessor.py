import os
import sys
from collections import Counter

import torch

def read_file(filename):
    '''read a conllx file and return sentences.
    
    Returns:
        - sentences: [s1, s2, s3, ...]
            sentences is a list which saves sentence, each sentence is as:
                sentence: [ids, forms, ... , heads, ...], the elements are not tokens, but fields:
                    ids: [1, 2, 3, ...], forms: [In, an, Oct., ...]

    '''

    start, sentences = 0, []

    # read all lines，and transform '\n' to ''
    with open(filename, 'r', encoding='UTF-8') as fr:
        lines = [line.strip() for line in fr]

    # split lines to sentences 
    for i, line in enumerate(lines):
        if not line: # whether a null string -> blank line
            # sentence consists of words from start to i-1 
            # sentence: [word, word, ...]
            # word: [id, form, ... , head, ...]
            sentence = [line.split('\t') for line in lines[start:i]]
            # sentence consists of fields, filed is a column 
            # sentence: [ids, forms, ... , heads, ...]
            # ids: (1, 2, 3, ...)
            # forms: (In, an, Oct., ...)
            sentence = [list(field) for field in zip(*sentence)]
            sentences.append(sentence)
            # updata start position
            start = i + 1

    return sentences

   
def create_vocab(sentences, vocab_file, special_labels, field, min_freq=2):
    '''build a vocabulary from training set.
    
    Params:
        - sentences: sentences from a conllx file
        - vocab_file: file store the vocabulary
        - special_labels: [<pad>, <unk>, <bos>]
        - field: token type of vocab, 'word' or 'char' or 'rel'
        - min_freq: drop out words whose frequency is less than it
    '''

    print('building a {} vocabulary !!!'.format(field))
    # which field of sentence
    if field == 'word':
        tokens = [word for s in sentences for word in s[1]]
    elif field == 'char':
        tokens = [char for s in sentences for word in s[1] for char in word]
    else: # rel
        tokens = [rel for s in sentences for rel in s[7]]

    # count words frequency
    counter = Counter(tokens) # counter: {word:count,...} ,
    pairs = [] # pairs: [(word, count), ...]
    # drop out words whose frequency is less than 'min_freq'
    for token, count in counter.items():
        if count >= min_freq:
            pairs.append((token, count))
    # leave words only
    tokens, _ = zip(*pairs) 
    # add 'padding word' and 'unknown word'
    tokens = special_labels + list(tokens) 

    print("size of {} vocabulary: {} !!!".format(field, len(tokens)))

    # write the vocabualry to a file, one word one line
    # whether dir exists
    path = vocab_file[0:vocab_file.rfind('/')]
    if not os.path.isdir(path):
        os.makedirs(path)
    with open(vocab_file, 'w', encoding='UTF-8') as fw: # 
        fw.write('\n'.join(tokens) + '\n')

def read_vocab(vocab_file):
    '''read the vocabulary from the file, return all words in the vocabulary.'''

    # print("reading the vocabulary !!!")

    with open(vocab_file, 'r', encoding='UTF-8') as fr:
        word_vocab = [word.strip() for word in fr.readlines()]

    # word_to_id = dict(zip(words, range(len(words))))

    return word_vocab


def load_pretrained_embedding(pretrained_embedding_file, embedding_dim, vocab_file, unk=None):
    '''load pretrained embeddings, such as Word2Vec.

    we will extend the vocabulary with pretrained words.
    vocabulary consists of two part: 
        words in training set | words in pretrained words but not in training set
    
    After extend the vocabulary, we will create a bigger embedding_matrix 
    based on the new vocabulary.
    all values are initialized to zero, and if a word is in pretrained word, 
    update the value with pretrained embedding, word only in training set are still zeros.
    at last, we normalize the matrix with std err.

    Params:
        - pretrained_embedding_file: the pretrained embeddings file
        - vocab_file: the file store the vocabulary
        - unk: unknown word in pretrained word, replace it with <unk> token

    Returns:
        - embedding_matrix: pretrained embeddings
            as a part of model, we don't need to save it by torch.save()
    '''
    
    print("load pretrained_embeddings!!!")

    word_vocab = read_vocab(vocab_file)

    # store word and embeddings 
    pretrained_words = [] # [word, ...]
    pretrained_embeddings = [] # [embeddings, ...]

    # read pretrained embeddings
    with open(pretrained_embedding_file, 'r', encoding='utf-8') as fr:
        i = 0
        for line in fr:
            i += 1
            row = line.strip().split(' ')  
            # the first line may contain the information of the file, ignore it  
            if len(row) < embedding_dim: 
                continue
            word = row[0]
            embedding = [float(i) for i in row[1:]]
            pretrained_words.append(word)
            pretrained_embeddings.append(embedding)

    # replace the unknown word in pretrained embeddings with unknown word in training set
    if unk is not None:
        unk_index = pretrained_words.index(unk)
        pretrained_words[unk_index] = word_vocab[1]

    # merge the training vocabulary and the total pretrained words
    word_vocab.extend(sorted(set(pretrained_words).difference(set(word_vocab))))
    word_to_id = {word:i for i, word in enumerate(word_vocab)}
    # update the  vocabualry, one word one line
    with open(vocab_file, 'w', encoding='UTF-8') as fw: # 
        fw.write('\n'.join(word_vocab) + '\n')

    # the matrix store the embeedings
    embedding_matrix = torch.zeros(len(word_vocab), embedding_dim) 

    indices = [word_to_id.get(word) for word in pretrained_words]
    # use values in pretrained
    embedding_matrix[indices] = torch.tensor(pretrained_embeddings)
    # matrix /= std err 
    embedding_matrix /= torch.std(embedding_matrix) 

    return embedding_matrix

# # TODO
# # get all dep_rel classes
# def get_tags(filename):
#     # print("所有文本类别：", end='')
#     tag_set = set()
#     with open(filename, 'r', encoding='UTF-8') as fr:
#         for line in fr:
#             tag_set.add(line.split('\t')[0])
    
#     tag_list = list(sorted(tag_set)) # 固定类别顺序
#     tag_to_id = dict(zip(tag_list, range(len(tag_list))))
#     # print(tag_list)

#     return tag_list, tag_to_id