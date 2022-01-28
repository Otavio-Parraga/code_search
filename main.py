import argparse
from pathlib import Path
from datasets import CodeSearchNetDataset
from models import CodeSearchModel
import pytorch_lightning as pl
from torch.utils.data import DataLoader


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-data', '--data_dir', type=str, default='datasets/Cleaned_CodeSearchNet/CodeSearchNet')
    parser.add_argument('-lan', '--language', type=str, default='javascript')

    # main configs
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    language = args.language

    # creating model
    #model = CodeSearchModel('Salesforce/codet5-base', '../code2test/pretrained_stuff')
    model = CodeSearchModel('microsoft/codebert-base', '../code2test/pretrained_stuff')

    # creating dataset
    train_dataset = CodeSearchNetDataset(data_dir / language / 'train.jsonl', model.tokenizer)
    val_dataset = CodeSearchNetDataset(data_dir / language / 'valid.jsonl', model.tokenizer)
    trainloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    valloader = DataLoader(val_dataset, batch_size=2, shuffle=False)

    # creating trainer
    trainer = pl.Trainer(gpus=1,
                         max_epochs=10,
                         gradient_clip_val=0.5,)
    trainer.fit(model, trainloader, valloader)
