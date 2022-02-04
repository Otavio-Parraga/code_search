# # TODO: obtain mrr
from torch.utils.data import DataLoader
from datasets import CodeSearchNetDataset
from models import CodeSearchModel
from pathlib import Path
from tqdm import tqdm
import torch
from metrics import mrr
from utils import dict_to_device


if __name__ == '__main__':
    data_dir = Path('datasets/CodeSearchNet')
    language = 'javascript'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # TODO: change to load a checkpoint
    model = CodeSearchModel.load_from_checkpoint(checkpoint_path='/home/parraga/projects/_masters/code_search/checkpoints/model_epoch=2.ckpt.ckpt',
                                                 model_name='microsoft/codebert-base', cache_path='../code2test/pretrained_stuff')
    val_dataset = CodeSearchNetDataset(
        data_dir / language / 'valid.jsonl', model.tokenizer)
    valloader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    model.to(device)
    model.eval()

    with torch.no_grad():
        queries, values, query_idxs, value_idxs = [], [], [], []
        for batch in tqdm(valloader):
            code, comment, url = batch
            code = dict_to_device(code, device)
            comment = dict_to_device(comment, device)
            value, query = model(code, comment)
            query_idxs.extend(url)
            value_idxs.extend(url)

            queries.append(query.cpu().data)
            values.append(value.cpu().data)

        queries = torch.cat(queries, dim=0)
        values = torch.cat(values, dim=0)

        print(mrr(queries.data.numpy(), values.data.numpy(), query_idxs, value_idxs))
