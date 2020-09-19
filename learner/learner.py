import sys

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR

from myparser.model import Model

def train(args, train_data_loader, dev_data_loader, test_data_loader=None, pretrained_embedding=None):
    '''训练模型
    
    Params:
        - train_data_loader: 
    
    '''

    model = Model(args)  # 模型
    model.load_pretrained(pretrained_embedding)
    print('the structure of Biaffine Parser is:\n', model)

    # TODO select correct Loss function
    criterion = nn.CrossEntropyLoss() 
    # Adam Optimizer
    optimizer = Adam(model.parameters(), args.learning_rate) 
    # learning rate decrease
    # new_lr = initial_lr * gamma**epoch = initial_lr * 0.75**(epoch/5000)
    scheduler = ExponentialLR(optimizer, args.decay**(1/args.decay_steps)) 

    best_epoch, best_accuracy = 0, 0
    for epoch in range(1, args.epochs+1):
        print('in epoch: {}'.format(epoch))
        # training mode，dropout is useful
        model.train() 
        total_loss = 0 
        for words, chars, heads, rels in train_data_loader:
            optimizer.zero_grad()

            # mask <bos> and <pad>
            mask = words.ne(args.pad_index)
            mask[:, 0] = 0
            # compute score
            # TODO whether handle rels
            socres_arc = model(words, chars) 
            loss = get_loss(criterion, socres_arc, heads, mask)
            # # compute grad
            # loss.backward() 
            # # clip grad which is larger than args.clip
            # nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            # # backpropagation
            # optimizer.step()
            # # updat learning rate
            # scheduler.step()
            total_loss += loss.item()
            print('total_loss: {}'.format(total_loss))
            sys.exit(0)
        print('total_loss: {}'.format(total_loss))

        accuracy = evaluate(model, dev_data_loader)
        print('Accuracy: {}'.format(accuracy))
        if accuracy > best_accuracy:
            best_epoch = epoch

        if epoch - best_epoch > args.patience:
            break

def evaluate(model, data_loader):
    '''评估模型性能'''
    
    model.eval()
    n_total, n_right = 0, 0 # 所有文本，分类对了的文本
    for (xs, ys), _ in data_loader:

        out = model(xs) # 计算log_softmax
        out = torch.argmax(out, dim=1)
        equal = torch.eq(out, ys)
        n_right += torch.sum(equal).item()
        n_total += len(ys)

    accuracy = n_right / n_total
    return accuracy

def get_loss(criterion, scores_arc, heads, mask):
    '''
    
    for score matrix, we drop out illegal tokens, that is <bos> and <pad>
    for each score vector, we just save scores that exceed sentence_len, but -inf will have no effect
    a score vector: [1., 4., 1., 0., 4., 3., 2., -inf]
    
    we split every token, not treat them as a part of sentences, that is we flat a batch of sentences 
    '''

    # (batch, seq_len, seq_len) -> (sum_of_sentences_len, seq_len)
    scores_arc = scores_arc[mask]
    # (batch, seq_len) -> sum_of_sentences_len
    heads = heads[mask]
    loss = criterion(scores_arc, heads)

    return loss

def decode(scores_arc, mask):
    pass


