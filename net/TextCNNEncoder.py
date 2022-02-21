# encoding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F

class TextCNNEncoder(nn.Module):
    def __init__(self, dropout):
        super(TextCNNEncoder, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.word_dims = 768

        # self.word_embed = nn.Embedding(vocab.word_size, self.word_dims, padding_idx=0)

        # extword_embed = vocab.load_pretrained_embs(word2vec_path)
        # extword_size, word_dims = extword_embed.shape
        # logging.info("Load extword embed: words %d, dims %d." % (extword_size, word_dims))
        #
        # self.extword_embed = nn.Embedding(extword_size, word_dims, padding_idx=0)
        # self.extword_embed.weight.data.copy_(torch.from_numpy(extword_embed))
        # self.extword_embed.weight.requires_grad = False

        input_size = self.word_dims

        self.filter_sizes = [2, 3, 4]  # n-gram window
        self.out_channel = 256 # 100
        self.convs = nn.ModuleList([nn.Conv2d(1, self.out_channel, (filter_size, input_size), bias=True)
                                    for filter_size in self.filter_sizes])

    def forward(self, batch_embed):
        # word_ids: sen_num x sent_len
        # extword_ids: sen_num x sent_len
        # batch_masks: sen_num x sent_len
        sen_num, sent_len, embedding_size = batch_embed.shape
        # print("sen_num:{}".format(sen_num))
        # print("sent_len:{}".format(sent_len))
        # print("embedding_size:{}".format(embedding_size))

        # word_embed = self.word_embed(word_ids)  # sen_num x sent_len x 100
        # extword_embed = self.extword_embed(extword_ids)
        # batch_embed = word_embed + extword_embed

        if self.training:
            batch_embed = self.dropout(batch_embed)

        batch_embed.unsqueeze_(1)  # sen_num x 1 x sent_len x 100

        pooled_outputs = []
        for i in range(len(self.filter_sizes)):
            filter_height = sent_len - self.filter_sizes[i] + 1
            conv = self.convs[i](batch_embed)
            hidden = F.relu(conv)  # sen_num x out_channel x filter_height x 1

            mp = nn.MaxPool2d((filter_height, 1))  # (filter_height, filter_width)
            pooled = mp(hidden).reshape(sen_num,
                                        self.out_channel)  # sen_num x out_channel x 1 x 1 -> sen_num x out_channel

            pooled_outputs.append(pooled)

        reps = torch.cat(pooled_outputs, dim=1)  # sen_num x total_out_channel

        if self.training:
            reps = self.dropout(reps)

        return reps