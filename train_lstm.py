# encoding=utf-8
import logging
import random
import numpy as np
import pandas as pd
import torch
from transformers import BasicTokenizer
from sklearn.model_selection import StratifiedKFold

from utils.vocab_utils import Vocab
from utils.trainer_utils import Trainer
from model import Model


logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(levelname)s: %(message)s')

# set seed
seed = 666
random.seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.manual_seed(seed)

# set cuda
gpu = 0
use_cuda = gpu >= 0 and torch.cuda.is_available()
if use_cuda:
    torch.cuda.set_device(gpu)
    device = torch.device("cuda", gpu)
else:
    device = torch.device("cpu")
logging.info("Use cuda: %s, gpu id: %d.", use_cuda, gpu)

# set tokenizer
# basic_tokenizer = BasicTokenizer()

# build word encoder
bert_path = "./bert_base_models_epoch3/"
save_model_path = 'bert.bin'
save_pred_path = "./bert.csv"
dropout = 0.15

# build sent encoder
sent_rep_size = 768 # word embedding size 词向量编码后的维度
sent_hidden_size = 768 # 编码句向量的隐藏层维度
sent_num_layers = 2

# set batch size
test_batch_size = 1 # 2 # 16
train_batch_size = 1 # 2 # 16

# set trainer
max_sent_len = 512
max_segment_num = 4 # 6

# k fold
n_fold = 5
sfolder = StratifiedKFold(n_splits=n_fold, random_state=None)


train_df = pd.read_csv('data/train_set.csv', sep='\t')#[:50000]
test_df = pd.read_csv('data/test_a.csv', sep='\t')
test_data = {'label': [0] * len(test_df), 'text': test_df['text'].to_list()}

train_text_folds = {}
train_label_folds = {}
test_text_folds = {}
test_label_folds = {}

for i, (train, test) in enumerate(sfolder.split(train_df['text'].to_list(), train_df['label'].to_list())):
    train_text_folds[i] = train_df['text'].values[train]
    train_label_folds[i] = train_df['label'].values[train]
    test_text_folds[i] = train_df['text'].values[test]
    test_label_folds[i] = train_df['label'].values[test]


test_pre_matrix = np.zeros((n_fold, len(test_data['label']), 14))  # 将5轮的测试概率分别保存起来
for i in range(n_fold):
    # 把效果不好的几个类别单独拿来训练看看
    cur_label = []
    cur_text = []
    for label, text in zip(list(train_label_folds[i]),train_text_folds[i]):
        r = random.randint(0, 9)
        if label == 4 or label == 5 or label == 7:
            if r <= 10: # 全部保留
                cur_label.append(label)
                cur_text.append(text)
        else:
            if r <= 2: # 4 只留一半
                cur_label.append(label)
                cur_text.append(text)
    train_data = {'label': cur_label, 'text': cur_text}
    # train_data = {'label': list(train_label_folds[i]), 'text': train_text_folds[i]}
    dev_data = {'label': list(test_label_folds[i]), 'text': test_text_folds[i]}
    datas = (train_data, dev_data, test_data)
    vocab = Vocab(train_data) # 为什么Vocab只用到train_data呢？？？
    model = Model(vocab, bert_path, max_sent_len, dropout,
                  sent_rep_size, sent_hidden_size, sent_num_layers, use_cuda, device)
    trainer = Trainer(model, vocab, datas, max_sent_len, max_segment_num,
                      train_batch_size, test_batch_size, use_cuda, device, save_model_path, save_pred_path)
    # trainer.train()
    test_logits = trainer.test()
    np.save("test_logits.npy",test_logits)
    test_pre_matrix[i, :, :] = test_logits
    break


test = np.argmax(test_pre_matrix.mean(axis=0), axis=1)
ans = pd.DataFrame(test)  # ans = pd.DataFrame(test,columns='label')
ans.to_csv('./bert_average_5.csv', index=False)


