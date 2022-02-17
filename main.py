from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from utils import dict_to_device, set_seed
from datasets import CodeSearchNetDataset
from torch.utils.data import DataLoader
from models import CodeSearchModel
import pytorch_lightning as pl
from pathlib import Path
from args import parse_args

if __name__ == '__main__':
    args = parse_args('training')

    set_seed()

    # main configs
    data_dir = Path(args.data_dir)
    language = args.language
    ptm = args.pretrained_model
    output_dir = Path(args.output_dir) / language / ptm.replace('/', '-')

    # creating model
    model = CodeSearchModel(ptm, '../code2test/pretrained_stuff')

    # creating dataset
    train_dataset = CodeSearchNetDataset(data_dir / language / 'train.jsonl', model.tokenizer)
    val_dataset = CodeSearchNetDataset(data_dir / language / 'valid.jsonl', model.tokenizer)
    trainloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    valloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir,
        filename='model_{epoch}',
        save_top_k=1,
        save_last=True,
        verbose=True,
        monitor='val_loss')

    early_stop_callback = EarlyStopping('val_loss', patience=3)

    # creating trainer
    trainer = pl.Trainer(gpus=2,
                         max_epochs=10,
                         gradient_clip_val=1,
                         strategy='ddp',
                         callbacks=[checkpoint_callback,
                                    early_stop_callback,
                                    ])
    trainer.fit(model, trainloader, valloader)
