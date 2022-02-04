# TODO: read csn dataset
import json
from torch.utils.data import Dataset
from utils import squeeze_dict


def read_jsonl(filepath):
    with open(filepath, 'r') as json_file:
        json_list = list(json_file)

    lines = []
    for json_str in json_list:
        result = json.loads(json_str)
        lines.append(result)
    return lines


class CodeSearchNetDataset(Dataset):
    def __init__(self, data_path, tokenizer):
        self.data = read_jsonl(data_path)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        instance = self.data[idx]
        code = instance["code"]
        comment = instance["docstring"]
        url = instance["url"]

        code = self.tokenizer(code, truncation=True, return_tensors='pt', padding='max_length')
        comment = self.tokenizer(comment, truncation=True, return_tensors='pt', padding='max_length')
        code = squeeze_dict(code)
        comment = squeeze_dict(comment)

        return code, comment, url
