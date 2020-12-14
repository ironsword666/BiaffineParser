
def mst():
    '''MST Algorithm parsing sentences to trees.
    
    
    '''
    pass

def eisner(scores_arc, mask):
    '''Eisner Algorithm parsing sentences to trees.

    We assign '-inf' to <pad>, resulting score of span contains <pad> impoosible to be max,
    that is to say, <pad> has no impact to final result,
    so, we can batchify the computation.

    Params:
        scores_arc (Tensor(batch_size, seq_len, seq_len)): with <bos> and <pad>
        mask (Tensor(batch_size,  seq_len)): position of <bos> and <bos> are False

    Returns:
        trees: 
    '''
    batch_size, seq_len, _ = scores_arc.shape
    # each sentence, score: Tensor(seq_len, seq_len), score[i,j] = j -> i
    for score in scores_arc:
        E = torch.zeros(seq_len, seq_len, 2, 2)
        T = torch.zeros(seq_len, seq_len, 2, 2)
        # m is the number of head-dep in a span, max(m) = seq_len - 1 
        for m in range(1, seq_len):
            # start of span
            for i in range(seq_len-m): 
                # end of span
                j = i + m
                # 
                e01 = E[i,i:j,1,0] + E[i+1:j+1,j,0,0] + score[i,j]
                E[i,j,0,1] = torch.max(e01)
                T[i,j,0,1] = torch.argmax(e01)
                # 
                e11 = E[i,i:j,1,0] + E[i+1:j+1,j,0,0] + score[j,i]
                E[i,j,1,1] = torch.max(e11)
                T[i,j,1,1] = torch.argmax(e11)
                # 
                e00 = E[i,i:j,0,0] + E[i:j,j,0,1]
                E[i,j,0,0] = torch.max(e00)
                T[i,j,0,0] = torch.argmax(e00)
                #
                e10 = E[i,i+1:j+1,1,1] + E[i+1:j+1,j,1,0]
                E[i,j,1,0] = torch.max(e10)
                T[i,j,1,0] = torch.argmax(e10)
        # print(E[0,seq_len-1,1,0])

def backtrack():
    pass

def crf(scores_arc, mask):
    '''compute partition Function.
    TODO compute the marginals.

    like eisner algorithm, but replace max() with sum()
    
     Params:
        scores_arc (Tensor(batch_size, seq_len, seq_len)): with <bos> and <pad>
        mask (Tensor(batch_size,  seq_len)): position of <bos> and <bos> are False

    Returns:
        logZ or Z (Tensor(batch_size)): 
    '''
    batch_size, seq_len, _ = scores_arc.shape
    # each sentence, score: Tensor(seq_len, seq_len), score[i,j] = j -> i
    for score in scores_arc:
        E = torch.zeros(seq_len, seq_len)
        # m is the number of head-dep in a span, max(m) = seq_len - 1 
        for m in range(1, seq_len):
            # start of span
            for i in range(seq_len-m): 
                # end of span
                j = i + m
                # 
                e01 = E[i,i:j] + E[i+1:j+1,j] + score[i,j]
                E[i,j] = torch.sum(e01)
        # print(E[0,seq_len-1,1,0])





