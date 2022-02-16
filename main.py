import argparse
from pathlib import Path
from metrics import mrr
from datasets import CodeSearchNetDataset
from models import CodeSearchModel
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch
from utils import dict_to_device
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

class MRRCallback(pl.Callback):
    def __init__(self, dataloader) -> None:
        self.dataloader = dataloader
    
    def on_epoch_start(self, trainer, pl_module):
        with torch.no_grad():
            pl_module.eval()
            mrr_value = 0
            queries, values, query_idxs, value_idxs = [], [], [], []
            for batch in self.dataloader:
                code, comment, url = batch
                value, query = model(code, comment)
                code = dict_to_device(code, pl_module.device)
                comment = dict_to_device(comment, pl_module.device)
                query_idxs.extend(url)
                value_idxs.extend(url)

                queries.append(query.cpu().data)
                values.append(value.cpu().data)

            queries = torch.cat(queries, dim=0)
            values = torch.cat(values, dim=0)

        mrr_value = mrr(queries.data.numpy(), values.data.numpy(), query_idxs, value_idxs)
        trainer.logger.log_metrics({"MRR": mrr_value})
        pl_module.train()


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
        filename='model_{epoch}',
        save_top_k=1,
        save_last=True,
        verbose=True,
        monitor='val_loss')

    early_stop_callback = EarlyStopping('val_loss', patience=3)

    # TODO: fix mrr callback
    mrr_callback = MRRCallback(valloader)

    # creating trainer
    trainer = pl.Trainer(gpus=2,
                         max_epochs=10,
                         gradient_clip_val=0.5,
                         strategy='ddp',
                         callbacks=[checkpoint_callback,
                                    early_stop_callback,
                                    #mrr_callback
                                    ])
    trainer.fit(model, trainloader, valloader)
