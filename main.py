from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from models import CodeSearchModel, load_tokenizer
from datasets import CodeSearchNetDataset
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from args import parse_args
from utils import set_seed
from pathlib import Path

if __name__ == '__main__':
    args = parse_args('training')

    set_seed()

    # main configs
    data_dir = Path(args.data_dir)
    language = args.language
    ptm = args.pretrained_model
    output_dir = Path(args.output_dir) / language / ptm.replace('/', '-')


    # creating tokenizer
    tokenizer = load_tokenizer(ptm, '../code2test/pretrained_stuff')

    # creating dataset
    train_dataset = CodeSearchNetDataset(data_dir / language / 'train.jsonl', tokenizer)
    val_dataset = CodeSearchNetDataset(data_dir / language / 'valid.jsonl', tokenizer)
    trainloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # creating model
    model = CodeSearchModel(ptm, '../code2test/pretrained_stuff', train_size=len(trainloader), epochs=args.epochs, scheduler=args.scheduler)

    # callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir,
        filename='best_model',
        save_top_k=1,
        save_last=True,
        verbose=True,
        monitor='val_loss')

    early_stop_callback = EarlyStopping('val_loss', patience=2)

    # creating trainer
    trainer = pl.Trainer(gpus=args.gpus,
                         max_epochs=args.epochs,
                         gradient_clip_val=1,
                         strategy='ddp',
                         callbacks=[checkpoint_callback,
                                    early_stop_callback,
                                    ])
    trainer.fit(model, trainloader, valloader)
