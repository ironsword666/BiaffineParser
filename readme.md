# 说明：
- 训练集和预训练词向量共用一个词汇表，但是使用不同的nn.Embedding()。只用对单词进行一次索引即可。
- 使用charlstm替代pos特征
- 

# 待完善的工作:
- numerilize()函数中tensor是否用Longtensor()?
- numerilize()函数中heads是否也要使用bos?
- numerilize()函数中chars是否写对了，一个char序列不用bos吗？也许可以不用。一个word整体作为一个tensor是否正确？

- Biaffine 为什么要 n_out 参数？因为还有rel-biaffine
- rel-biaffine为什么两边都要填充？

- CrossEntropyLoss需要的得分矩阵，与Dozat论文中的好像需要转置？通过mask得到的不需要
  

