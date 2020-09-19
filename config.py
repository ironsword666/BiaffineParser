import os
import sys

class Config(object):
    '''神经网络的配置参数'''

    # [网络]
    max_len = 50             # 文本的最大长度
    
    embedding_dim = 100          # 词向量的维度
    num_layers = 3             # layers of bilstm
    n_lstm_hidden = 400       # lstm的hidden层
    n_mlp_arc =  500            # mlp的hidden层

    min_freq = 2                # 低频词频率
    fix_len = 20            # 每个单词最多有多少个字符

    # is_pretrained = False         
    # is_static = False         
    # word_nums = 50000          
    
    pad = '<pad>'
    unk = '<unk>'
    bos = '<bos>'
    unk_pretrained = '' # unkown word in pretrained embeddings

    pad_index = 0
    unk_index = 1
    bos_index = 2

    # [优化系数]
    learning_rate = 1e-3        # 学习率
    # l2_reg_lambda = 0.01        # 正则化系数
    mu = 0.9
    nu = 0.9
    epsilon = 1e-12
    clip = 5              # 梯度截断上限
    decay = 0.75              # 学习率下降
    decay_steps = 5000

    # [训练设置]
    batch_size = 50             # 每批训练的大小
    epochs = 30            # 总迭代轮次
    patience = 5           # dev上多少次不提升则停止训练

    # [数据]
    base_dir = './data/ptb/'                   # 数据文件所在的目录
    train_file = os.path.join(base_dir, 'train.conllx')     # 训练集
    dev_file = os.path.join(base_dir, 'dev.conllx')         # 验证集
    test_file = os.path.join(base_dir, 'test.conllx')       # 测试集
    # word2vec_file = './data/embedding/glove.6B.100d.txt' # word2vec

    model_file = './save/model/model_save'
    vocab_file = './save/vocab/word_vocabulary'                # 词汇表
    char_vocab_file = './save/vocab/char_vocabulary'                # 字符词汇表
    rel_vocab_file = './save/vocab/rel_vocabulary'              # 关系词汇表
    # pretrained_embedding_file = './save/embedding/pretrained.pt'  # word2vec数据部分

    # Comand
    # python run.py --word2vec_file=./data/embedding/glove.6B.100d.txt --unk=unk