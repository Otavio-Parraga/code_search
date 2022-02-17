import argparse

def parse_args(mode='training'):
    parser = argparse.ArgumentParser()
    parser.add_argument('-data', '--data_dir', type=str, default='datasets/CodeSearchNet')
    parser.add_argument('-lan', '--language', type=str, default='javascript')
    parser.add_argument('-ptm', '--pretrained_model', type=str, default='microsoft/codebert-base')

    if mode == 'training':
        return training_args(parser)

    elif mode == 'evaluation':
        return evaluation_args(parser)
    
    return parser.parse_args()

def training_args(parser):
    parser.add_argument('-out', '--output_dir', type=str, default='checkpoints')
    return parser.parse_args()

def evaluation_args(parser):
    parser.add_argument('-ckpt', '--checkpoint_path', type=str, default=None)
    return parser.parse_args()