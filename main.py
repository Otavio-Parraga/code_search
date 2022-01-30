import argparse
from pathlib import Path
from datasets import CodeSearchNetDataset
from models import CodeSearchModel
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-data', '--data_dir', type=str, default='datasets/CodeSearchNet')
    parser.add_argument('-lan', '--language', type=str, default='javascript')
    parser.add_argument('-out', '--output_dir', type=str, default='checkpoints')

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
    trainloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    valloader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    # callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.output_dir,
        filename='model_{epoch}.ckpt',
        save_top_k=3,
        save_last=True,
        verbose=True,
        monitor='val_loss')

    early_stop_callback = EarlyStopping('val_loss', patience=3)

    # creating trainer
    trainer = pl.Trainer(gpus=2,
                         max_epochs=10,
                         gradient_clip_val=0.5,
                         strategy='ddp',
                         callbacks=[checkpoint_callback,
                                    early_stop_callback])
    trainer.fit(model, trainloader, valloader)
