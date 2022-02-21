# encoding=utf-8
import torch
import torch.nn as nn
import logging
import numpy as np

from net.LSTMEncoder import LSTMEncoder
from net.BertEncoder import WordBertEncoder
from net.Attention import Attention
from net.TextCNNEncoder import TextCNNEncoder

class Model(nn.Module):
    def __init__(self, vocab, bert_path, max_sent_len, dropout,
                 sent_rep_size, sent_hidden_size, sent_num_layers, use_cuda, device):
        super(Model, self).__init__()
        self.sent_rep_size = sent_rep_size # 256
        self.doc_rep_size = sent_hidden_size * 2
        self.all_parameters = {}
        parameters = []
        self.word_encoder = WordBertEncoder(bert_path, max_sent_len, dropout)
        bert_parameters = self.word_encoder.get_bert_parameters()

        self.sent_encoder = LSTMEncoder(self.sent_rep_size, sent_hidden_size, sent_num_layers, dropout)
        self.sent_attention = Attention(self.doc_rep_size)
        parameters.extend(list(filter(lambda p: p.requires_grad, self.sent_encoder.parameters())))
        parameters.extend(list(filter(lambda p: p.requires_grad, self.sent_attention.parameters())))

        self.out = nn.Linear(self.doc_rep_size, vocab.label_size, bias=True)
        parameters.extend(list(filter(lambda p: p.requires_grad, self.out.parameters())))

        if torch.cuda.is_available():
            device = torch.device("cuda")
            self.to(device)
        else:
            device = torch.device("cpu")
            self.to(device)

        if len(parameters) > 0:
            self.all_parameters["basic_parameters"] = parameters
        self.all_parameters["bert_parameters"] = bert_parameters

        logging.info('Build model with bert word encoder, lstm sent encoder.')

        para_num = sum([np.prod(list(p.size())) for p in self.parameters()]) # 运行到这后，cuda内存拉爆了，把别的代码弄崩了
        logging.info('Model param num: %.2f M.' % (para_num / 1e6))

    def forward(self, batch_inputs):
        # batch_inputs(batch_inputs1, batch_inputs2): b x doc_len x sent_len
        # batch_masks : b x doc_len x sent_len
        batch_inputs1, batch_inputs2, batch_masks = batch_inputs
        batch_size, max_doc_len, max_sent_len = batch_inputs1.shape[0], batch_inputs1.shape[1], batch_inputs1.shape[2]
        batch_inputs1 = batch_inputs1.view(batch_size * max_doc_len, max_sent_len)  # sen_num x sent_len
        batch_inputs2 = batch_inputs2.view(batch_size * max_doc_len, max_sent_len)  # sen_num x sent_len
        batch_masks = batch_masks.view(batch_size * max_doc_len, max_sent_len)  # sen_num x sent_len

        sent_reps = self.word_encoder(batch_inputs1, batch_inputs2)  # sen_num x sent_rep_size

        sent_reps = sent_reps.view(batch_size, max_doc_len, self.sent_rep_size)  # b x doc_len x sent_rep_size
        batch_masks = batch_masks.view(batch_size, max_doc_len, max_sent_len)  # b x doc_len x max_sent_len
        sent_masks = batch_masks.bool().any(2).float()  # b x doc_len

        sent_hiddens = self.sent_encoder(sent_reps, sent_masks)  # b x doc_len x doc_rep_size
        doc_reps, atten_scores = self.sent_attention(sent_hiddens, sent_masks)  # b x doc_rep_size

        batch_outputs = self.out(doc_reps)  # b x num_labels

        return batch_outputs


class TextCNNModel(nn.Module):
    def __init__(self, vocab, bert_path, max_sent_len, dropout,
                 sent_rep_size, sent_hidden_size, sent_num_layers, use_cuda, device):
        super(TextCNNModel, self).__init__()
        self.sent_rep_size = 768  # 256
        self.doc_rep_size = sent_hidden_size * 2
        self.all_parameters = {}
        parameters = []
        self.word_encoder = WordBertEncoder(bert_path, max_sent_len, dropout)
        bert_parameters = self.word_encoder.get_bert_parameters()

        self.textcnn_encoder = TextCNNEncoder(dropout)
        parameters.extend(list(filter(lambda p: p.requires_grad, self.textcnn_encoder.parameters())))

        self.sent_encoder = LSTMEncoder(self.sent_rep_size, sent_hidden_size, sent_num_layers, dropout)
        self.sent_attention = Attention(self.sent_rep_size)  # Attention(self.doc_rep_size) LSTM是doc_rep_size
        parameters.extend(list(filter(lambda p: p.requires_grad, self.sent_encoder.parameters())))
        parameters.extend(list(filter(lambda p: p.requires_grad, self.sent_attention.parameters())))

        self.out = nn.Linear(self.sent_rep_size, vocab.label_size, bias=True)  # self.doc_rep_size (LSTM)
        parameters.extend(list(filter(lambda p: p.requires_grad, self.out.parameters())))

        if use_cuda:
            self.to(device)

        if len(parameters) > 0:
            self.all_parameters["basic_parameters"] = parameters
        self.all_parameters["bert_parameters"] = bert_parameters

        logging.info('Build model with bert word encoder, lstm sent encoder.')

        para_num = sum([np.prod(list(p.size())) for p in self.parameters()])
        logging.info('Model param num: %.2f M.' % (para_num / 1e6))

    def forward(self, batch_inputs):
        # batch_inputs(batch_inputs1, batch_inputs2): b x doc_len x sent_len
        # batch_masks : b x doc_len x sent_len
        batch_inputs1, batch_inputs2, batch_masks = batch_inputs
        batch_size, max_doc_len, max_sent_len = batch_inputs1.shape[0], batch_inputs1.shape[1], batch_inputs1.shape[2]
        # shape:[batch_size_ , seq_length]
        batch_inputs1 = batch_inputs1.view(batch_size * max_doc_len, max_sent_len)  # sen_num x sent_len
        batch_inputs2 = batch_inputs2.view(batch_size * max_doc_len, max_sent_len)  # sen_num x sent_len
        batch_masks = batch_masks.view(batch_size * max_doc_len, max_sent_len)  # sen_num x sent_len

        # sent_reps = self.word_encoder(batch_inputs1, batch_inputs2)  # sen_num x sent_rep_size 当pooled=True
        sent_reps = self.word_encoder(batch_inputs1, batch_inputs2)  # sen_num x seq_len x sent_rep_size
        # print("sent_reps1:{}".format(sent_reps.shape)) # sent_reps1:torch.Size([1, 281, 768])
        sent_reps = self.textcnn_encoder(sent_reps)
        # print("sent_reps2:{}".format(sent_reps.shape)) # sent_reps2:torch.Size([1, 768])

        sent_reps = sent_reps.view(batch_size, max_doc_len, self.sent_rep_size)  # b x doc_len x sent_rep_size
        batch_masks = batch_masks.view(batch_size, max_doc_len, max_sent_len)  # b x doc_len x max_sent_len
        sent_masks = batch_masks.bool().any(2).float()  # b x doc_len

        # 下面这三行是使用了LSTM来作为句子的编码方式
        # sent_hiddens = self.sent_encoder(sent_reps, sent_masks)  # b x doc_len x doc_rep_size
        # doc_reps, atten_scores = self.sent_attention(sent_hiddens, sent_masks)  # b x doc_rep_size
        # batch_outputs = self.out(doc_reps)  # b x num_labels

        # 下面这三行是使用了TextCNN来作为句子的编码方式
        # sen_num, sent_len, embedding_size TextCNN输入需要的格式
        # sent_hiddens = self.textcnn_encoder(sent_reps)  # b x doc_len x doc_rep_size
        # 上面这个textcnn貌似还缺乏一个mask
        doc_reps, atten_scores = self.sent_attention(sent_reps, sent_masks)  # b x doc_rep_size
        batch_outputs = self.out(doc_reps)  # b x num_labels

        # 下面这两行是直接使用bert后的向量进行attention（暂未成功）
        # doc_reps, atten_scores = self.sent_attention(sent_reps, sent_masks)  # b x doc_rep_size
        # batch_outputs = self.out(doc_reps)  # b x num_labels

        return batch_outputs
