import pytorch_lightning as pl
from pathlib import Path
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch
from transformers import T5Model, RobertaTokenizer, AutoModel, AutoTokenizer
from utils import cosine_sim

mt_pairs = {'Salesforce/codet5-base': {'model': T5Model, 'tokenizer': RobertaTokenizer},
            'microsoft/codebert-base': {'model': AutoModel, 'tokenizer': AutoTokenizer},
            'microsoft/graphcodebert-base': {'model': AutoModel, 'tokenizer': AutoTokenizer}}


def load_model_and_tokenizer(model_name, cache_path):
    cache_path = Path(cache_path)
    cache_path.mkdir(exist_ok=True, parents=True)
    if model_name in mt_pairs.keys():
        model = mt_pairs[model_name]['model'].from_pretrained(model_name, cache_dir=cache_path)
        tokenizer = mt_pairs[model_name]['tokenizer'].from_pretrained(model_name, cache_dir=cache_path)
    else:
        model = AutoModel.from_pretrained(model_name, cache_dir='./pretrained_stuff')
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir='./pretrained_stuff')
    return model, tokenizer


class BertEncoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.bert_encoder = model

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        embedding = self.bert_encoder(
            input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
        )

        embedding = embedding[1]
        embedding = F.normalize(embedding, p=2, dim=1)
        return embedding


class T5Encoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.t5_encoder = model

    def forward(self, input_ids, attention_mask):
        embedding = self.t5_encoder.encoder(
            input_ids=input_ids, attention_mask=attention_mask,
        )

        # TODO: change this
        embedding = torch.mean(embedding.last_hidden_state, dim=1)
        embedding = F.normalize(embedding, p=2, dim=1)
        return embedding


class CodeSearchModel(pl.LightningModule):
    def __init__(self, model_name, cache_path='./pretrained_stuff'):
        super(CodeSearchModel, self).__init__()
        model, self.tokenizer = load_model_and_tokenizer(model_name, cache_path)
        self.encoder = T5Encoder(model) if 't5' in model_name else BertEncoder(model)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, code, comment):
        code = self.encoder(**code)
        comment = self.encoder(**comment)
        return code, comment

    def training_step(self, batch, batch_idx):
        code, comment = batch
        encoded_code, encoded_comment = self(code, comment)
        scores = cosine_sim(encoded_comment, encoded_code)
        loss = self.criterion(scores, torch.arange(encoded_code.size(0), device=scores.device))
        self.log('train_loss', loss)
        return loss

    #def validation_step(self, batch, batch_idx):
    #    code, comment = batch
    #    encoded_code, encoded_comment = self(code, comment)
    #    #scores = torch.einsum("ab,cb->ac", encoded_comment, encoded_code)
    #    scores = cosine_sim(encoded_comment, encoded_code)
    #    loss = self.criterion(scores, torch.arange(encoded_code.size(0), device=scores.device))
    #    self.log('val_loss', loss)
    #    return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        return optimizer
