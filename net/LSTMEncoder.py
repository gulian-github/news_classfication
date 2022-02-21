# encoding=utf-8
import torch.nn as nn

class LSTMEncoder(nn.Module):
    def __init__(self, sent_rep_size, sent_hidden_size, sent_num_layers, dropout):
        '''
        LSTM编码器，用于对句子的编码(含MASK)
        输入：[batch_size, sequence_len, sent_rep_size]
        输出：[batch_size, sequence_len, sent_hidden_size*2]
        :param sent_rep_size: int, 输入句子的embedding size
        :param sent_hidden_size: int, LSTM的隐藏层的size
        :param sent_num_layers: int, LSTM的层数
        :param dropout: float，dropout的比例
        '''
        super(LSTMEncoder, self).__init__()
        self.dropout = nn.Dropout(dropout)

        self.sent_lstm = nn.LSTM(
            input_size=sent_rep_size,
            hidden_size=sent_hidden_size,
            num_layers=sent_num_layers,
            batch_first=True,
            bidirectional=True
        )

    def forward(self, sent_reps, sent_masks):
        # sent_reps:  b x doc_len x sent_rep_size
        # sent_masks: b x doc_len

        sent_hiddens, _ = self.sent_lstm(sent_reps)  # b x doc_len x hidden*2
        sent_hiddens = sent_hiddens * sent_masks.unsqueeze(2)

        if self.training:
            sent_hiddens = self.dropout(sent_hiddens)

        return sent_hiddens