# encoding=utf-8
from transformers import BertModel
import torch.nn as nn
import logging

class WordBertEncoder(nn.Module):
    def __init__(self,bert_path, max_sent_len, dropout):
        super(WordBertEncoder, self).__init__()
        self.dropout = nn.Dropout(dropout)

        self.tokenizer = WhitespaceTokenizer(bert_path,max_sent_len)
        self.bert = BertModel.from_pretrained(bert_path)

        self.pooled = True # 如果是textcnn需要改成False
        logging.info('Build Bert encoder with pooled {}.'.format(self.pooled))

    def encode(self, tokens):
        tokens = self.tokenizer.tokenize(tokens)
        return tokens

    def get_bert_parameters(self):
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_parameters = [
            {'params': [p for n, p in self.bert.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.01},
            {'params': [p for n, p in self.bert.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
        return optimizer_parameters

    def forward(self, input_ids, token_type_ids):
        # input_ids: sen_num x bert_len
        # token_type_ids: sen_num  x bert_len

        # sen_num x bert_len x 256, sen_num x 256
        # sequence_output, pooled_output = self.bert(input_ids=input_ids, token_type_ids=token_type_ids)
        outputs = self.bert(input_ids=input_ids, token_type_ids=token_type_ids)
        sequence_output, pooled_output = outputs[0], outputs[1]

        if self.pooled:
            reps = pooled_output
        else:
            reps = sequence_output # [:, 0, :]  # sen_num x 256

        if self.training:
            reps = self.dropout(reps)

        return reps


class WhitespaceTokenizer():
    """WhitespaceTokenizer with vocab."""

    def __init__(self,bert_path,max_sent_len):
        vocab_file = bert_path + 'vocab.txt'
        self._token2id = self.load_vocab(vocab_file)
        self._id2token = {v: k for k, v in self._token2id.items()}
        self.max_len = max_sent_len
        self.unk = 1

        logging.info("Build Bert vocab with size %d." % (self.vocab_size))

    def load_vocab(self, vocab_file):
        f = open(vocab_file, 'r')
        lines = f.readlines()
        lines = list(map(lambda x: x.strip(), lines))
        vocab = dict(zip(lines, range(len(lines))))
        return vocab

    def tokenize(self, tokens):
        assert len(tokens) <= self.max_len - 2
        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        output_tokens = self.token2id(tokens)
        return output_tokens

    def token2id(self, xs):
        if isinstance(xs, list):
            return [self._token2id.get(x, self.unk) for x in xs]
        return self._token2id.get(xs, self.unk)

    @property
    def vocab_size(self):
        return len(self._id2token)