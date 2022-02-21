# encoding=utf-8
import time
import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score, precision_score, recall_score
import pandas as pd

from utils.data_utils import *
from utils.optim_utils import Optimizer
from utils.adversarial_utils import FGM

class Trainer():
    def __init__(self, model, vocab, datas, max_sent_len, max_segment,
                 train_batch_size, test_batch_size, use_cuda, device, save_model_path, save_pred_path):
        self.model = model
        self.report = True

        # count
        self.step = 0
        self.early_stop = -1
        self.best_train_f1, self.best_dev_f1 = 0, 0
        self.epochs = 1 # 3 # 20
        self.clip = 5.0
        self.early_stops = 3
        self.log_interval = 50
        self.last_epoch = self.epochs

        # others
        self.use_cuda = use_cuda
        self.device = device
        self.save_model_path = save_model_path
        self.save_pred_path = save_pred_path

        # batch size
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size

        train_data, dev_data, test_data = datas
        self.train_data = get_examples(train_data, model.word_encoder, vocab, max_sent_len, max_segment)
        self.batch_num = int(np.ceil(len(self.train_data) / float(self.train_batch_size)))
        self.dev_data = get_examples(dev_data, model.word_encoder, vocab, max_sent_len, max_segment)
        self.test_data = get_examples(test_data, model.word_encoder, vocab, max_sent_len, max_segment)

        # criterion
        self.criterion = nn.CrossEntropyLoss()

        # label name
        self.target_names = vocab.target_names

        # optimizer
        self.optimizer = Optimizer(model.all_parameters, steps=self.batch_num * self.epochs)


    def train(self):
        logging.info('Start training...')
        # 加载已经训练过的模型
        self.model.load_state_dict(torch.load(self.save_model_path))
        for epoch in range(1, self.epochs + 1):
            train_f1 = self._train(epoch)

            dev_f1, _ = self._eval(epoch) # 验证集的p_pred暂时不用

            if self.best_dev_f1 <= dev_f1:
                logging.info(
                    "Exceed history dev = %.2f, current dev = %.2f" % (self.best_dev_f1, dev_f1))
                torch.save(self.model.state_dict(), self.save_model_path)

                self.best_train_f1 = train_f1
                self.best_dev_f1 = dev_f1
                self.early_stop = 0
            else:
                self.early_stop += 1
                if self.early_stop == self.early_stops:
                    logging.info(
                        "Eearly stop in epoch %d, best train: %.2f, dev: %.2f" % (
                            epoch - self.early_stops, self.best_train_f1, self.best_dev_f1))
                    self.last_epoch = epoch
                    break

    def test(self):
        self.model.load_state_dict(torch.load(self.save_model_path))
        f1, logits = self._eval(self.last_epoch + 1, test=True)
        return logits

    def _train(self, epoch):
        self.optimizer.zero_grad()
        self.model.train()

        start_time = time.time()
        epoch_start_time = time.time()
        overall_losses = 0
        losses = 0
        batch_idx = 1
        y_pred = []
        y_true = []
        for batch_data in data_iter(self.train_data, self.train_batch_size, shuffle=True):
            torch.cuda.empty_cache()
            batch_inputs, batch_labels = self.batch2tensor(batch_data)
            batch_outputs = self.model(batch_inputs)
            loss = self.criterion(batch_outputs, batch_labels)
            loss.backward()

            # # 对抗验证
            # fgm = FGM(self.model)
            # fgm.attack()  # embedding被修改了
            # # optimizer.zero_grad() # 如果不想累加梯度，就把这里的注释取消
            # loss_sum = self.criterion(self.model(batch_inputs), batch_labels)
            # loss_sum.backward()  # 反向传播，在正常的grad基础上，累加对抗训练的梯度
            # fgm.restore()  # 恢复Embedding的参数

            loss_value = loss.detach().cpu().item()
            losses += loss_value
            overall_losses += loss_value

            y_pred.extend(torch.max(batch_outputs, dim=1)[1].cpu().numpy().tolist())
            y_true.extend(batch_labels.cpu().numpy().tolist())

            nn.utils.clip_grad_norm_(self.optimizer.all_params, max_norm=self.clip)
            for optimizer, scheduler in zip(self.optimizer.optims, self.optimizer.schedulers):
                optimizer.step()
                scheduler.step()
            self.optimizer.zero_grad()

            self.step += 1

            if batch_idx % self.log_interval == 0:
                elapsed = time.time() - start_time

                lrs = self.optimizer.get_lr()
                logging.info(
                    '| epoch {:3d} | step {:3d} | batch {:3d}/{:3d} | lr{} | loss {:.4f} | s/batch {:.2f}'.format(
                        epoch, self.step, batch_idx, self.batch_num, lrs,
                        losses / self.log_interval,
                        elapsed / self.log_interval))

                losses = 0
                start_time = time.time()

            batch_idx += 1

        overall_losses /= self.batch_num
        during_time = time.time() - epoch_start_time

        # reformat
        overall_losses = self.reformat(overall_losses, 4)
        score, f1 = self.get_score(y_true, y_pred)

        logging.info(
            '| epoch {:3d} | score {} | f1 {} | loss {:.4f} | time {:.2f}'.format(epoch, score, f1,
                                                                                  overall_losses,
                                                                                  during_time))
        if set(y_true) == set(y_pred) and self.report:
            report = classification_report(y_true, y_pred, digits=4, target_names=self.target_names)
            logging.info('\n' + report)

        return f1

    def _eval(self, epoch, test=False):
        self.model.eval()
        start_time = time.time()
        data = self.test_data if test else self.dev_data
        y_pred = []
        y_true = []
        logits = []
        with torch.no_grad():
            for batch_data in data_iter(data, self.test_batch_size, shuffle=False):
                torch.cuda.empty_cache()
                batch_inputs, batch_labels = self.batch2tensor(batch_data)
                batch_outputs = self.model(batch_inputs)
                y_pred.extend(torch.max(batch_outputs, dim=1)[1].cpu().numpy().tolist()) # test_num
                y_true.extend(batch_labels.cpu().numpy().tolist()) # test_num
                logits.extend(batch_outputs.cpu().numpy().tolist()) # test_num * label_size

            score, f1 = self.get_score(y_true, y_pred)

            df = pd.DataFrame({'label': y_pred})
            df.to_csv(self.save_pred_path, index=False, sep=',')

            during_time = time.time() - start_time

            logging.info(
                '| epoch {:3d} | dev | score {} | f1 {} | time {:.2f}'.format(epoch, score, f1,
                                                                                  during_time))
            if set(y_true) == set(y_pred) and self.report:
                report = classification_report(y_true, y_pred, digits=4, target_names=self.target_names)
                logging.info('\n' + report)

        return f1, logits

    def batch2tensor(self, batch_data):
        '''
            [[label, doc_len, [[sent_len, [sent_id0, ...], [sent_id1, ...]], ...]]
        '''
        batch_size = len(batch_data)
        doc_labels = []
        doc_lens = []
        doc_max_sent_len = []
        for doc_data in batch_data:
            doc_labels.append(doc_data[0])
            doc_lens.append(doc_data[1])
            sent_lens = [sent_data[0] for sent_data in doc_data[2]]
            max_sent_len = max(sent_lens)
            doc_max_sent_len.append(max_sent_len)

        max_doc_len = max(doc_lens)
        max_sent_len = max(doc_max_sent_len)

        batch_inputs1 = torch.zeros((batch_size, max_doc_len, max_sent_len), dtype=torch.int64)
        batch_inputs2 = torch.zeros((batch_size, max_doc_len, max_sent_len), dtype=torch.int64)
        batch_masks = torch.zeros((batch_size, max_doc_len, max_sent_len), dtype=torch.float32)
        batch_labels = torch.LongTensor(doc_labels)

        for b in range(batch_size):
            for sent_idx in range(doc_lens[b]):
                sent_data = batch_data[b][2][sent_idx]
                for word_idx in range(sent_data[0]):
                    batch_inputs1[b, sent_idx, word_idx] = sent_data[1][word_idx]
                    batch_inputs2[b, sent_idx, word_idx] = sent_data[2][word_idx]
                    batch_masks[b, sent_idx, word_idx] = 1

        if torch.cuda.is_available(): # self.use_cuda: 不太完善
            device = torch.device("cuda")
            batch_inputs1 = batch_inputs1.to(device)
            batch_inputs2 = batch_inputs2.to(device)
            batch_masks = batch_masks.to(device)
            batch_labels = batch_labels.to(device)
            # batch_inputs1 = batch_inputs1.to(self.device)
            # batch_inputs2 = batch_inputs2.to(self.device)
            # batch_masks = batch_masks.to(self.device)
            # batch_labels = batch_labels.to(self.device)

        return (batch_inputs1, batch_inputs2, batch_masks), batch_labels

    def get_score(self,y_ture, y_pred):
        y_ture = np.array(y_ture)
        y_pred = np.array(y_pred)
        f1 = f1_score(y_ture, y_pred, average='macro') * 100
        p = precision_score(y_ture, y_pred, average='macro') * 100
        r = recall_score(y_ture, y_pred, average='macro') * 100

        return str((self.reformat(p, 2), self.reformat(r, 2), self.reformat(f1, 2))), self.reformat(f1, 2)

    def reformat(self,num, n):
        return float(format(num, '0.' + str(n) + 'f'))