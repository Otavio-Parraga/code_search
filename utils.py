import torch.nn.functional as F

def squeeze_dict(dict):
    return {k: v.squeeze(0) for k, v in dict.items()}

def cosine_sim(matrix_a, matrix_b):
    """Build cosine similarity matrix."""
    matrix_a = F.normalize(matrix_a, p=2, dim=1)
    matrix_b = F.normalize(matrix_b, p=2, dim=1)
    return matrix_a.mm(matrix_b.t())