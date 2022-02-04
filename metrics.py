# TODO: implement MRR

import numpy as np


def mrr(query_embed, base_embed, query_index, base_index):
    """This functions comes from graphcode bert"""

    scores = np.matmul(query_embed,base_embed.T)

    sort_ids = np.argsort(scores, axis=-1, kind='quicksort', order=None)[:,::-1]

    ranks=[]
    for url, sort_id in zip(query_index,sort_ids):
        rank=0
        find=False
        for idx in sort_id[:1000]:
            if find is False:
                rank+=1
            if base_index[idx]==url:
                find=True
        if find:
            ranks.append(1/rank)
        else:
            ranks.append(0)

    result = {
        "mrr": float(np.mean(ranks))
    }
    return result