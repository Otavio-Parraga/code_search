ckpt_path='/home/parraga/projects/_masters/code_search/checkpoints'
model='microsoft-graphcodebert-base'
ptm='microsoft/graphcodebert-base'

CUDA_VISIBLE_DEVICES=2 python main.py -lang javascript -ptm $ptm --gpus 1
lang='javascript'
CUDA_VISIBLE_DEVICES=2 python evaluation.py -ckpt $ckpt_path/$lang/$model/best_model.ckpt -lang $lang -ptm $ptm

CUDA_VISIBLE_DEVICES=2 python main.py -lang go -ptm $ptm --gpus 1
lang='go'
CUDA_VISIBLE_DEVICES=2 python evaluation.py -ckpt $ckpt_path/$lang/$model/best_model.ckpt -lang $lang -ptm $ptm

CUDA_VISIBLE_DEVICES=2 python main.py -lang java -ptm $ptm --gpus 1
lang='java'
CUDA_VISIBLE_DEVICES=2 python evaluation.py -ckpt $ckpt_path/$lang/$model/best_model.ckpt -lang $lang -ptm $ptm

CUDA_VISIBLE_DEVICES=2 python main.py -lang php -ptm $ptm --gpus 1
lang='php'
CUDA_VISIBLE_DEVICES=2 python evaluation.py -ckpt $ckpt_path/$lang/$model/best_model.ckpt -lang $lang -ptm $ptm

CUDA_VISIBLE_DEVICES=2 python main.py -lang ruby -ptm $ptm --gpus 1
lang='ruby'
CUDA_VISIBLE_DEVICES=2 python evaluation.py -ckpt $ckpt_path/$lang/$model/best_model.ckpt -lang $lang -ptm $ptm

CUDA_VISIBLE_DEVICES=2 python main.py -lang python -ptm $ptm --gpus 1
lang='python'
CUDA_VISIBLE_DEVICES=2 python evaluation.py -ckpt $ckpt_path/$lang/$model/best_model.ckpt -lang $lang -ptm $ptm
