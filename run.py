import argparse

from config import Config
from utils.preprocessor import read_file, create_vocab, read_vocab, load_pretrained_embedding
from utils.dataloader import TextDataSet, batchify
from learner.learner import train 

from myparser.model import Model


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Biaffine Parser')

    parser.add_argument('--word2vec_file', default=None, 
                        help='the path to pretrained embeddings file') 
    parser.add_argument('--unk', default=None,
                        help='what is the unknown word in pretrained embeddings') 
    args = parser.parse_args()

    config = Config()
    
    # TODO updata arguments
    config.word2vec_file = args.word2vec_file
    config.unk_pretrained = args.unk

    # rename
    args = config

    # get training set
    train_sentences = read_file(args.train_file)
    # drop out sentence whose length is larger than 'max_len'
    train_sentences = [sentence for sentence in train_sentences if len(sentence[0]) <= args.max_len]
    # build the vocabulary
    special_labels = [args.pad, args.unk, args.bos]
    create_vocab(train_sentences, args.vocab_file, special_labels, 'word', args.min_freq)
    # words number in training set
    args.word_nums = len(read_vocab(args.vocab_file))

    # load pretrained embeddings
    if args.word2vec_file:
        pretrained_embedding = load_pretrained_embedding(args.word2vec_file, args.embedding_dim, args.vocab_file, args.unk_pretrained)
    else:
        pretrained_embedding = None

    word_vocab = read_vocab(args.vocab_file)

    # create char vocab
    create_vocab(train_sentences, args.char_vocab_file, special_labels, 'char', args.min_freq)
    char_vocab = read_vocab(args.char_vocab_file)
    args.char_nums = len(char_vocab)
    # create rel vocab 
    create_vocab(train_sentences, args.rel_vocab_file, special_labels, 'rel', min_freq=1)
    rel_vocab = read_vocab(args.rel_vocab_file)

    # TODO complete the TextDataset
    train_data = TextDataSet(args, train_sentences, word_vocab, char_vocab, rel_vocab)
    train_data_loader = batchify(train_data, args.batch_size, shuffle=True)
    print("create train_data_loader successfully !!!")

    dev_sentences = read_file(args.dev_file)
    dev_data = TextDataSet(args, dev_sentences, word_vocab, char_vocab, rel_vocab)
    dev_data_loader = batchify(dev_data, args.batch_size, shuffle=False)
    print("create dev_data_loader successfully !!!")

    test_sentences = read_file(args.test_file)
    test_data = TextDataSet(args, test_sentences, word_vocab, char_vocab, rel_vocab)
    test_data_loader = batchify(test_data, args.batch_size, shuffle=True)
    print("create test_data_loader successfully !!!")

    train(args, train_data_loader, dev_data_loader, test_data_loader, pretrained_embedding)


