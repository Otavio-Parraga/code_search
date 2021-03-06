ckpt_path='/home/parraga/projects/_masters/code_search/checkpoints'
model='Salesforce-codet5-base'
ptm='Salesforce/codet5-base'

CUDA_VISIBLE_DEVICES=1 python main.py -lang javascript -ptm Salesforce/codet5-base --gpus 1 -bs 16
lang='javascript'
CUDA_VISIBLE_DEVICES=1 python evaluation.py -ckpt $ckpt_path/$lang/$model/best_model.ckpt -lang $lang -ptm $ptm

CUDA_VISIBLE_DEVICES=1 python main.py -lang go -ptm Salesforce/codet5-base --gpus 1 -bs 16
lang='go'
CUDA_VISIBLE_DEVICES=1 python evaluation.py -ckpt $ckpt_path/$lang/$model/best_model.ckpt -lang $lang -ptm $ptm

CUDA_VISIBLE_DEVICES=1 python main.py -lang java -ptm Salesforce/codet5-base --gpus 1 -bs 16
lang='java'
CUDA_VISIBLE_DEVICES=1 python evaluation.py -ckpt $ckpt_path/$lang/$model/best_model.ckpt -lang $lang -ptm $ptm

CUDA_VISIBLE_DEVICES=1 python main.py -lang php -ptm Salesforce/codet5-base --gpus 1 -bs 16
lang='php'
CUDA_VISIBLE_DEVICES=1 python evaluation.py -ckpt $ckpt_path/$lang/$model/best_model.ckpt -lang $lang -ptm $ptm

CUDA_VISIBLE_DEVICES=1 python main.py -lang ruby -ptm Salesforce/codet5-base --gpus 1 -bs 16
lang='ruby'
CUDA_VISIBLE_DEVICES=1 python evaluation.py -ckpt $ckpt_path/$lang/$model/best_model.ckpt -lang $lang -ptm $ptm

CUDA_VISIBLE_DEVICES=1 python main.py -lang python -ptm Salesforce/codet5-base --gpus 1 -bs 16
lang='python'
CUDA_VISIBLE_DEVICES=1 python evaluation.py -ckpt $ckpt_path/$lang/$model/best_model.ckpt -lang $lang -ptm $ptm