import torch.nn.functional as F
import numpy as np
import torch
import random
import os


def squeeze_dict(dict):
    return {k: v.squeeze(0) for k, v in dict.items()}

def cosine_sim(matrix_a, matrix_b):
    """Build cosine similarity matrix."""
    matrix_a = F.normalize(matrix_a, p=2, dim=1)
    matrix_b = F.normalize(matrix_b, p=2, dim=1)
    return matrix_a.mm(matrix_b.t())

def dict_to_device(dict, device):
    return {k: v.to(device) for k, v in dict.items()}

def set_seed(seed=42):

    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True