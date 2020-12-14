import sys
import os
from datetime import datetime, timedelta

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR


from src.models.biaffine_parser import BiaffineParser
from src.utils.dataloader import TextDataSet, TextDataLoader
from src.utils.corpus import Conll, Embedding
from src.utils.metric import Metric
from src.utils.common import bos_token, eos_token, pad_token, unk_token
from src.utils.field import Field, SubWordField

class Parser(object):

    def __init__(self, args, fields, model):
        '''
        Args:
            args (Config):
            fields:
            model:
        '''
        self.args = args
        # self.true_fields = {k: value[0] for k, value in tri_fields.items()}
        # # {field:alias, ...}
        # self.alias_fields = {value[0]: value[1] for value in tri_fields.values()}
        self.fields = fields
        # field to name in Corpus
        # TODO HEAD and ARC field
        self.fields_alias = {
            fields['WORD']: Conll.FIELD_NAMES[1],
            fields['FEAT']: Conll.FIELD_NAMES[1],
            # fields['POS']: Conll.FIELD_NAMES[3]
        }
        self.parser_model = model

    def train(self, args):
        # TODO no need to use args, as args is assigned to __init__

        # build dataset
        train = TextDataSet(Conll.load(args.ftrain), self.fields_alias)
        train.build_loader(batch_size=args.batch_size, shuffle=True)
        dev = TextDataSet(Conll.load(args.fdev), self.fields_alias)
        dev.build_loader(batch_size=args.batch_size)
        test = TextDataSet(Conll.load(args.ftest), self.fields_alias)
        test.build_loader(batch_size=args.batch_size)

        self.criterion = nn.CrossEntropyLoss(reduction='mean') 
        # Adam Optimizer
        self.optimizer = Adam(self.parser_model.parameters(), args.lr) 
        # learning rate decrease
        # new_lr = initial_lr * gamma**epoch = initial_lr * 0.75**(epoch/5000)
        self.scheduler = ExponentialLR(self.optimizer, args.decay**(1/args.decay_steps))

        total_time = timedelta()
        best_epoch, metric = 0, Metric()
        for epoch in range(1, args.epochs+1):
            start_time = datetime.now()

            print('training epoch {} :'.format(epoch))
            loss, metric = self.train_epoch(args, train.data_loader)
            print('train loss: {}'.format(loss))
            accuracy = self.evaluate(args, dev.data_loader)
            print('dev accuracy: {}'.format(accuracy))

            time_diff = datetime.now() - start_time
            print('epoch time: {}'.format(time_diff))
            total_time += time_diff
            # if accuracy > best_accuracy:
                # best_epoch = epoch

            # if epoch - best_epoch > args.patience:
                # break
        accuracy = self.evaluate(args, test.data_loader)
        print('test accuracy: {}'.format(accuracy))
        print('total_time: {}'.format(total_time))

    def train_epoch(self, args, data_loader):
        '''
        Args:
            args:
            data_loader:
        '''
        self.parser_model.train() 
        total_loss = 0 

        for words, chars, heads, rels in data_loader:

            self.optimizer.zero_grad()

            # mask <bos> and <pad>
            mask = words.ne(args.pad_index)
            # mask = words.ne(self.fields['WORD'].pad_index) & words.ne(self.fields['WORD'].bos_index) & words.ne(self.fields['WORD'].eos_index)

            mask[:, 0] = 0
            # compute score
            # TODO whether handle rels
            socres_arc = self.parser_model(words, chars) 
            loss =  self.get_loss(socres_arc, heads, mask)
            # compute grad
            loss.backward() 
            # clip grad which is larger than args.clip
            nn.utils.clip_grad_norm_(self.parser_model.parameters(), args.clip)
            # backpropagation
            self.optimizer.step()
            # updat learning rate
            self.scheduler.step()
            total_loss += loss.item()
            print('total_loss: {}'.format(total_loss))
            sys.exit(0)

        # TODO metric    
        
        return total_loss / len(data_loader), 0

    @torch.no_grad()
    def evaluate(self, args, data_loader):
        ''''''
        
        self.parser_model.eval()
       
        for words, feats, heads, rels in data_loader:

            # mask = words.ne(self.fields['WORD'].pad_index) & words.ne(self.fields['WORD'].bos_index) & words.ne(self.fields['WORD'].eos_index)
            scores = self.parser_model(words, feats) 

        uas = 0

        return uas
         
        
    def get_loss(self, scores, heads, mask, use_crf=False):
        '''
        local loss: use cross-entropy and scores
        global loss: use crf and scores
        
        for score matrix, we drop out illegal tokens, that is <bos> and <pad>
        for each score vector, we still save scores that exceed sentence_len, but -inf will have no effect
        a score vector: [1., 4., 1., 0., 4., 3., 2., -inf]
        
        we split every token, not treat them as a part of sentences, that is we flat a batch of sentences.

        Params:
            scores (Tensor(batch, seq_len, tag_nums)): ...
            # tags (Tensor(batch, seq_len)): ...
            mask (Tensor(batch, seq_len)): mask <bos> <eos > and <pad>
            crf: whether use crf to calculate loss
        '''

        # # TODO complete
        # if not use_crf:
        #     # (batch, seq_len, tag_nums) -> (sum_of_sentences_len, tag_nums)
        #     scores = scores[mask]
        #     # (batch, seq_len) -> sum_of_sentences_len
        #     target = tags[mask]
        #     loss = criterion(scores, target)
        # else:
        #     loss = crf(scores, heads, mask)
        #     if torch.isnan(loss):
        #         raise Exception('loss is nan')
        # return loss
        pass


    def decode(self, scores_arc, mask):
        pass

    def save(self, path):
        ''' save parser to path '''
        pass
    
    @classmethod
    def build_parser(cls, args):
        
        # directory to save model and fields
        if not os.path.isdir(args.save_dir):
            os.makedirs(args.save_dir)
        args.parser_fields = os.path.join(args.save_dir, 'parser_fields')
        args.parser_model = os.path.join(args.save_dir, 'parser_model')
        # TODO merge fields and model in a torch.save
        # build tagger fields
        if not os.path.exists(args.parser_fields):
            print('Create fields for Tagger !')
            WORD = Field(pad_token=pad_token, unk_token=unk_token, bos_token=bos_token, 
                         eos_token=eos_token, lower=True)
            # TODO char-bilstm, use eos_token
            FEAT = SubWordField(pad_token=pad_token, unk_token=unk_token, bos_token=bos_token,
                                eos_token=eos_token, fix_len=args.fix_len, tokenize=list)
            # TODO need bos_token and eos_token?
            POS = Field(bos_token=bos_token, eos_token=eos_token)
            conll = Conll.load(args.ftrain)

            fields = {
                'WORD': WORD,
                'FEAT': FEAT,
                'POS': POS
            }
            
            # field.build_vocab(getattr(conll, name), (Embedding.load(args.w2v, args.unk) if args.w2v else None))
            WORD.build_vocab(examples=getattr(conll, Conll.FIELD_NAMES[1]), 
                             min_freq=args.min_freq, 
                             embed=(Embedding.load(args.w2v) if args.w2v else None))
            FEAT.build_vocab(examples=getattr(conll, Conll.FIELD_NAMES[1]))
            POS.build_vocab(examples=getattr(conll, Conll.FIELD_NAMES[3]))
        # TODO load fields
        else:
            pass

        # build tagger model
        # # TODO
        # args.update({
        #     'n_words': WORD.vocab.n_init,
        # })
        # parser_model = cls.MODEL(**args)
        # TODO
        parser_model = BiaffineParser(n_words=WORD.vocab.n_init,
                                      n_chars=FEAT.vocab.n_init,
                                      n_tags=POS.vocab.n_init,
                                      n_embed=args.n_embed,
                                      n_char_embed=args.n_char_embed,
                                      n_feat_embed=args.n_feat_embed,
                                      n_lstm_hidden=args.n_lstm_hidden,
                                      n_lstm_layer=args.n_lstm_layer,
                                      pad_index=WORD.pad_index,
                                      unk_index=WORD.unk_index)

        # TODO save model
        if os.path.exists(args.parser_model):
            state = torch.load(args.parser_model, map_location=args.device)
            parser_model.load_pretrained(state['pretrained'])
            parser_model.load_state_dict(state['state_dict'], False)
        
        parser_model.to(args.device)
        
        print('The network structure of POS Tagger is:\n', parser_model)

        
        return cls(args, fields, parser_model)

    


