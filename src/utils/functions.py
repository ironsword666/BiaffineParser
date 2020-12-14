# -*- coding: UTF-8 -*-

def preprocessing_heads(example):
    '''
    Params:
        example (list[str]): a list of strings representing heads of words, 
            such as ['2', '3', '0', '3', '3'] 
    
    Returns:
        ret: [2, 3, 0, 3, 3] 
    '''
    return [int(i) for i in example]