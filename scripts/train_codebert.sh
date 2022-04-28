ckpt_path='/home/parraga/projects/_masters/code_search/checkpoints'
model='microsoft-codebert-base'
ptm='microsoft/codebert-base'

CUDA_VISIBLE_DEVICES=1 python main.py -lang javascript --gpus 1
lang='javascript'
CUDA_VISIBLE_DEVICES=1 python evaluation.py -ckpt $ckpt_path/$lang/$model/best_model.ckpt -lang $lang -ptm $ptm

CUDA_VISIBLE_DEVICES=1 python main.py -lang go --gpus 1
lang='go'
CUDA_VISIBLE_DEVICES=1 python evaluation.py -ckpt $ckpt_path/$lang/$model/best_model.ckpt -lang $lang -ptm $ptm

CUDA_VISIBLE_DEVICES=1 python main.py -lang java --gpus 1
lang='java'
CUDA_VISIBLE_DEVICES=1 python evaluation.py -ckpt $ckpt_path/$lang/$model/best_model.ckpt -lang $lang -ptm $ptm

CUDA_VISIBLE_DEVICES=1 python main.py -lang php --gpus 1
lang='php'
CUDA_VISIBLE_DEVICES=1 python evaluation.py -ckpt $ckpt_path/$lang/$model/best_model.ckpt -lang $lang -ptm $ptm

CUDA_VISIBLE_DEVICES=1 python main.py -lang ruby --gpus 1
lang='ruby'
CUDA_VISIBLE_DEVICES=1 python evaluation.py -ckpt $ckpt_path/$lang/$model/best_model.ckpt -lang $lang -ptm $ptm

CUDA_VISIBLE_DEVICES=1 python main.py -lang python --gpus 1
lang='python'
CUDA_VISIBLE_DEVICES=1 python evaluation.py -ckpt $ckpt_path/$lang/$model/best_model.ckpt -lang $lang -ptm $ptm
