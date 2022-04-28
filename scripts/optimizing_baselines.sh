ckpt_path='/home/parraga/projects/_masters/code_search/test_optimizations/sgd'
model='microsoft-codebert-base'
ptm='microsoft/codebert-base'
lang='javascript'

CUDA_VISIBLE_DEVICES=3 python main.py -lang $lang -out test_optimizations/sgd --gpus 1

CUDA_VISIBLE_DEVICES=3 python evaluation.py -ckpt $ckpt_path/$lang/$model/best_model.ckpt -lang $lang -ptm $ptm


ckpt_path='/home/parraga/projects/_masters/code_search/test_optimizations/sgd'
model='microsoft-codebert-base'
ptm='microsoft/codebert-base'
lang='ruby'

CUDA_VISIBLE_DEVICES=3 python main.py -lang $lang -out test_optimizations/sgd --gpus 1

CUDA_VISIBLE_DEVICES=3 python evaluation.py -ckpt $ckpt_path/$lang/$model/best_model.ckpt -lang $lang -ptm $ptm