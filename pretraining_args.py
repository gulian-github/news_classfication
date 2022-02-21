# -----------ARGS---------------------
pretrain_train_path = "data/corpus.txt" # "./sentence.txt"
pretrain_dev_path = "data/corpus.txt" # "./sentence.txt"
bert_config_json = "bert_models/config.json"
vocab_file = "bert_models/vocab.txt"
output_dir = "outputs"

max_seq_length = 512 # 128
do_train = True
do_lower_case = True
train_batch_size = 4
eval_batch_size = 4
learning_rate = 1e-4
num_train_epochs = 1 # 20
warmup_proportion = 0.1
no_cuda = False
local_rank = -1
seed = 42
gradient_accumulation_steps = 1
fp16 = False
loss_scale = 0.
masked_lm_prob = 0.15
max_predictions_per_seq = 80 #20
