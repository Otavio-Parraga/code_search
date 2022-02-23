ckpt_path='/home/parraga/projects/_masters/code_search/checkpoints'
model='microsoft-codebert-base'
ptm='microsoft/codebert-base'

lang='javascript'
CUDA_VISIBLE_DEVICES=0 python evaluation.py -ckpt $ckpt_path/$lang/$model/model_epoch=1.ckpt -lang $lang -ptm $ptm

lang='go'
CUDA_VISIBLE_DEVICES=0 python evaluation.py -ckpt $ckpt_path/$lang/$model/model_epoch=0.ckpt -lang $lang -ptm $ptm

lang='java'
CUDA_VISIBLE_DEVICES=0 python evaluation.py -ckpt $ckpt_path/$lang/$model/model_epoch=0.ckpt -lang $lang -ptm $ptm

lang='php'
CUDA_VISIBLE_DEVICES=0 python evaluation.py -ckpt $ckpt_path/$lang/$model/model_epoch=1.ckpt -lang $lang -ptm $ptm

lang='ruby'
CUDA_VISIBLE_DEVICES=0 python evaluation.py -ckpt $ckpt_path/$lang/$model/model_epoch=0-v1.ckpt -lang $lang -ptm $ptm

lang='python'
CUDA_VISIBLE_DEVICES=0 python evaluation.py -ckpt $ckpt_path/$lang/$model/best_model.ckpt -lang $lang -ptm $ptm
