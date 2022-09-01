# ------------ IMPORTS --------------------
import configparser
import os
import random
import torch
import tensorflow as tf
import pycuda.driver as cuda
import wandb

from datetime import datetime
from pathlib import Path
from tensorflow.python.data.experimental import AutoShardPolicy
from transformers import BertTokenizerFast, TFLongformerModel, LongformerConfig, LongformerModel

# ------------ CONFIGURATION AND CONSTANTS --------------------

# the following line shall not be necessary since it is set by MetaCentrum scheduler
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"


tf.config.set_visible_devices([], 'GPU')

config = configparser.ConfigParser()
config.read('config.ini')
LOCAL = False
if 'local' in config:
    LOCAL = True

print("CUDA: ", torch.cuda.is_available())

BATCH_SIZE_PER_REPLICA = 32
SEQ_LEN = 1024
ATTENTION_WINDOW = 256

VOCAB_SIZE = 50265

KEEP_CKPT_EVERY_HOUR = 6
MAX_TO_KEEP_CKPT = 3
SAVE_EVERY_BATCH = 500
LR_EVERY_BATCH_UPDATE = 5

DATASET_SEQ_LEN = 1024
DATASET_SIZE = 218_500_000
EXAMPLES_FOR_1024 = 21_000_000
PREVIOUSLY_PROCESSED_EXAMPLES = 147_728_128

BASE_SHARED_FOLDER = "/storage/brno3-cerit/home/pasekj/code-bert"
DATASET_CKPT = "last_processed_batch.log"
CKPT_DIR = 'tf_ckpts'
DATASET_TF_RECORDS = "data"

os.chdir(BASE_SHARED_FOLDER)

if LOCAL:
    BATCH_SIZE_PER_REPLICA = 2
    SEQ_LEN = 9
    ATTENTION_WINDOW = 8

TOKENIZER_FOLDER = "tokenizer/wpt_cased_50k"
tokenizer = None

workers_count = 1

EPOCH_SIZE_PER_REPLICA = DATASET_SIZE / workers_count
BATCH_SIZE = BATCH_SIZE_PER_REPLICA * workers_count
PREFETCH_BATCHES = 100
BUFFER_SIZE = BATCH_SIZE * PREFETCH_BATCHES

WARMUP = 240000
INIT_LR = .00001

my_id = 0
my_local_id = 0
torch.cuda.set_device(0)

if torch.cuda.is_available():
    dev = f"cuda:0"
    torch.cuda.set_device(torch.cuda.current_device())
    cuda.init()
    print("using CUDA device: ", torch.cuda.current_device(), " ", cuda.Device(torch.cuda.current_device()).name())
else:
    dev = "cpu"

SCRATCH_DIR = os.environ["SCRATCH"]


class DatasetLoader():
    def __init__(self, workers_count, my_id, skip_examples, scratch):
        self.skip_examples_total = skip_examples
        self.skip_examples = int(skip_examples / workers_count)
        self.workers_count = workers_count
        self.my_id = my_id
        random.seed(666)  # seed from hell
        tf.random.set_seed(666)  # seed from hell for TF dataset

        data_path = os.path.join(scratch, DATASET_TF_RECORDS)
        print(f"{my_id}: Getting data from {data_path}")
        self.paths_tfrecord_qa = \
            [str(x) for x in Path(data_path).glob("*QT_AT*.tfrecords")] + \
            [str(x) for x in Path(data_path).glob("*QC_AC*.tfrecords")] + \
            [str(x) for x in Path(data_path).glob("*QT_AC*.tfrecords")] + \
            [str(x) for x in Path(data_path).glob("*QC_AT*.tfrecords")]

        self.paths_tfrecord_sp = \
            [str(x) for x in Path(data_path).glob("*AC_AT*.tfrecords")] + \
            [str(x) for x in Path(data_path).glob("*QC_QT*.tfrecords")]

        # print(f"Loaded QA: {len(self.paths_tfrecord_qa)} and SP: {len(self.paths_tfrecord_sp)}")
        # print(f"Whole dataset : {len(self.paths_tfrecord_qa+self.paths_tfrecord_sp)} \n\t {self.paths_tfrecord_qa+self.paths_tfrecord_sp}")
        random.shuffle(self.paths_tfrecord_qa)
        random.shuffle(self.paths_tfrecord_sp)

        def filterPaths(ex):
            # i, v = ex
            # if i % workers_count == my_id:
            #     return True
            # else:
            #     return False
            return True

        self.filtered_paths_tfrecord_qa = [ex for i, ex in list(filter(filterPaths, enumerate(self.paths_tfrecord_qa)))]
        self.filtered_paths_tfrecord_sp = [ex for i, ex in list(filter(filterPaths, enumerate(self.paths_tfrecord_sp)))]

    def set_base_folder(self, path):
        def map_base_folder(ex):
            fname = os.path.basename(ex)
            return os.path.join(path, fname)

        self.filtered_paths_tfrecord_qa = list(map(map_base_folder, self.filtered_paths_tfrecord_qa))
        self.filtered_paths_tfrecord_sp = list(map(map_base_folder, self.filtered_paths_tfrecord_sp))

    @tf.function
    def build_masked_lm_data(self, input_ids, mask_id, predict_prob, mask_predicted_prob, perm_prob, num_tokens):
        output = tf.identity(input_ids)
        predict_mask = tf.squeeze(
            tf.random.categorical(tf.math.log([[1 - predict_prob, predict_prob]]), tf.shape(input_ids)[0]))
        output = output * predict_mask - 100 * (1 - predict_mask)
        mask_mask = tf.squeeze(
            tf.random.categorical(tf.math.log([[1 - mask_predicted_prob, mask_predicted_prob]]),
                                  tf.shape(input_ids)[0]))
        input_ids = input_ids * (1 - predict_mask * mask_mask) + mask_id * predict_mask * mask_mask
        replace_mask = tf.squeeze(
            tf.random.categorical(tf.math.log([[1 - perm_prob, perm_prob]]), tf.shape(input_ids)[0]))
        input_ids = input_ids * (1 - predict_mask * (1 - mask_mask) * replace_mask) + tf.random.uniform(
            tf.shape(input_ids),
            maxval=num_tokens,
            dtype=tf.int64) * predict_mask * (1 - mask_mask) * replace_mask
        return input_ids, output

    def get_dataset(self, epoch_size_per_replica):
        _skip_batches = self.skip_examples // BATCH_SIZE_PER_REPLICA
        print(f"..skipping {_skip_batches}/{epoch_size_per_replica // BATCH_SIZE_PER_REPLICA}")
        while _skip_batches > epoch_size_per_replica:
            _skip_batches -= epoch_size_per_replica

        # print(
        #     f"WORKER_{self.my_id} of {self.workers_count} is taking {len(self.filtered_paths_tfrecord_qa + self.filtered_paths_tfrecord_sp)}"
        #     f"...list:{self.filtered_paths_tfrecord_qa + self.filtered_paths_tfrecord_sp}")
        qa_dataset = self.get_tf_dataset(self.filtered_paths_tfrecord_qa, [0, 1])
        sp_dataset = self.get_tf_dataset(self.filtered_paths_tfrecord_qa, [1, 0])

        qa_sp_dataset = tf.data.experimental.sample_from_datasets([qa_dataset, sp_dataset])
        qa_sp_dataset = qa_sp_dataset.skip(_skip_batches)
        qa_sp_dataset = qa_sp_dataset.prefetch(PREFETCH_BATCHES)

        return qa_sp_dataset

    def get_tf_dataset(self, paths, task_mask):
        train_dataset = tf.data.TFRecordDataset(paths)

        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = AutoShardPolicy.OFF
        train_dataset = train_dataset.with_options(options)

        feature_description = {
            'ids': tf.io.FixedLenSequenceFeature([], tf.int64, default_value=0, allow_missing=True),
            'att_mask': tf.io.FixedLenFeature([], tf.int64, default_value=0),
            'token_type': tf.io.FixedLenFeature([], tf.int64, default_value=0),
            'nsp': tf.io.FixedLenFeature([], tf.int64, default_value=0),
        }

        def _parse_function(example_proto):
            parsed_example = tf.io.parse_single_example(example_proto, feature_description)
            return parsed_example

        train_dataset = train_dataset.map(_parse_function)

        def _unpack_dataset(example, task_mask):
            att_mask_vec = tf.ones([tf.maximum(example['att_mask'], SEQ_LEN)], dtype=tf.int32)
            glob_loc_mask = tf.concat([tf.ones([1], dtype=tf.int32), tf.zeros(example['att_mask'] - 1, dtype=tf.int32)],
                                      axis=0)

            token_type_vec = tf.concat([tf.zeros([example['token_type']], dtype=tf.int32), tf.ones(
                [example['att_mask'] - tf.cast(example['token_type'], dtype=tf.int64)], dtype=tf.int32)], axis=0)
            inputs, outputs = self.build_masked_lm_data(example['ids'], 4, 0.15, 0.80, 0.10, len(tokenizer))
            task_mask = tf.constant(task_mask, dtype=tf.int64)
            qa_sp_lables = tf.ones(shape=[2], dtype=tf.int64) * example['nsp']
            # return (inputs[:SEQ_LEN], att_mask_vec[:SEQ_LEN], token_type_vec[:SEQ_LEN]), (outputs[:SEQ_LEN], qa_sp_lables,task_mask)
            return (inputs[:SEQ_LEN], att_mask_vec[:SEQ_LEN], glob_loc_mask[:SEQ_LEN], token_type_vec[:SEQ_LEN]), (
                outputs[:SEQ_LEN], qa_sp_lables, task_mask)

        train_dataset = train_dataset.map(lambda x: _unpack_dataset(x, task_mask))
        train_dataset = train_dataset.shuffle(BUFFER_SIZE)
        train_dataset = train_dataset.padded_batch(BATCH_SIZE_PER_REPLICA,
                                                   # padded_shapes=(([SEQ_LEN], [SEQ_LEN], [SEQ_LEN]), ([SEQ_LEN], [2],[2])),
                                                   padded_shapes=(
                                                       ([SEQ_LEN], [SEQ_LEN], [SEQ_LEN], [SEQ_LEN]),
                                                       ([SEQ_LEN], [2], [2])),
                                                   padding_values=(
                                                       (
                                                           tf.constant(tokenizer.pad_token_id, dtype=tf.int64),
                                                           tf.constant(0, dtype=tf.int32),
                                                           tf.constant(0, dtype=tf.int32),
                                                           tf.constant(0, dtype=tf.int32)
                                                       ), (
                                                           tf.constant(0, dtype=tf.int64),
                                                           tf.constant(0, dtype=tf.int64),
                                                           tf.constant(0, dtype=tf.int64)
                                                       )

                                                   ))
        return train_dataset


def export_for_publishing(train_dataset, model):
    if my_id != 0:
        return
    path = os.path.join(BASE_SHARED_FOLDER, "model")
    if not os.path.exists(path):
        os.makedirs(path)
    print("..saving last checkpoint for publishing", end="")
    for examples, labels in train_dataset:
        example = [torch.tensor(e.numpy(), dtype=torch.int64).cuda(dev) for e in examples]
        model(example)
        model.get_longformer().save_pretrained(path)
        tokenizer.save_pretrained(path)
        print("...saved")
        exit(0)


def get_scrach():
    # worker scratch directory
    scratch = os.path.join(SCRATCH_DIR, f"worker{my_local_id}")
    return scratch


def train():
    global tokenizer
    print("..cluster established")

    try:
        with open(DATASET_CKPT, "r", encoding="utf8") as dataset_ckpt_file:
            start_from = int(dataset_ckpt_file.read())
    except:
        start_from = 0

    if my_id == 0:
        configuration = LongformerConfig(vocab_size=VOCAB_SIZE, attention_window=ATTENTION_WINDOW,
                                         max_position_embeddings=1024 + 2).to_dict()
        configuration["max_seq_len"] = SEQ_LEN
        configuration["batch_size"] = BATCH_SIZE
        wandb.init(project='code-bert-pretraining',
                   name='Pretraining 1024',
                   save_code=True,
                   reinit=True,
                   resume='allow',
                   id='0000031',
                   config=configuration)

    worker_scratch = get_scrach()
    dl = DatasetLoader(workers_count, my_id, start_from - PREVIOUSLY_PROCESSED_EXAMPLES, worker_scratch)
    tokenizer = BertTokenizerFast(os.path.join(TOKENIZER_FOLDER, "vocab.txt"), strip_accents=False)
    train_dataset = dl.get_dataset(EPOCH_SIZE_PER_REPLICA)

    current_time = datetime.now().strftime("%Y-%m-%d---%H-%M")
    try:
        cuda_dev_num = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
    except:
        cuda_dev_num = "unk"
    train_log_dir = f"{current_time}--w{workers_count}x{cuda_dev_num}-b{BATCH_SIZE_PER_REPLICA}"
    train_log_dir = os.path.join(BASE_SHARED_FOLDER, "logs", train_log_dir)

    if my_id == 0:
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)

    # lr = tf.Variable(INIT_LR)
    # optimizer = tf.keras.optimizers.Adam(learning_rate=lr, clipvalue=4.0, clipnorm=2.0)

    class TBCallback(tf.keras.callbacks.Callback):
        def on_batch_end(self, batch, logs=None):
            if batch % 100 == 0:
                print(logs)
            if my_id == 0:
                with train_summary_writer.as_default():
                    with tf.name_scope("losses"):
                        tf.summary.scalar('loss', logs["loss"], step=batch)
                        tf.summary.scalar('loss_qa', logs["loss_qa"], step=batch)
                        tf.summary.scalar('loss_sp', logs["loss_sp"], step=batch)
                        tf.summary.scalar('loss_mlm', logs["output_1_loss"], step=batch)
                        tf.summary.scalar('loss_qa_sp', logs["output_2_loss"], step=batch)
                    with tf.name_scope("optimizer"):
                        print(optimizer.param_groups[0]['lr'])
                        logs["lr"] = optimizer.param_groups[0]['lr']
                        logs["batch_size"] = BATCH_SIZE
                        logs["max_seq_len"] = SEQ_LEN
                        tf.summary.scalar('LR', optimizer.param_groups[0]['lr'], step=batch)
                wandb.log(logs, step=batch)

    class LRScheduler(tf.keras.callbacks.Callback):

        def get_lr(self, batch):
            if batch < WARMUP:  # warm-up
                return INIT_LR * batch / (WARMUP)
            else:  # decay
                return INIT_LR * (1 - ((batch - EXAMPLES_FOR_1024) / DATASET_SIZE))

        def on_batch_end(self, batch, logs=None):
            if my_id == 0:
                print(f"LR: {self.get_lr(batch)}")
            if (batch // BATCH_SIZE) % LR_EVERY_BATCH_UPDATE == 0:
                optimizer.param_groups[0]['lr'] = self.get_lr(batch)

    class SaveCkpt(tf.keras.callbacks.Callback):
        def __init__(self):
            pass

        def on_batch_end(self, batch, logs=None):
            if my_id == 0:
                if (batch // BATCH_SIZE) % SAVE_EVERY_BATCH == 0 and batch != 0:
                    dataset_ckp_path = os.path.join(BASE_SHARED_FOLDER, DATASET_CKPT)
                    with open(dataset_ckp_path, "w", encoding="utf8") as dataset_ckpt:
                        to_write = batch
                        print(f"saving progress {to_write}")
                        dataset_ckpt.write(f"{to_write}")

                    path = os.path.join(BASE_SHARED_FOLDER, CKPT_DIR, f"{batch}.ckpt")
                    torch.save({
                        'model_state_dict': net.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }, path)

                    print(
                        f"\nSaving checkpoint(ckpt,dataset) for batch:{batch // BATCH_SIZE} ==> sent:{batch} from own shard:"
                        f" {path} ", end="")

                # print("saving histograms ...", end="")
                # with train_summary_writer.as_default():
                #     for w in model.trainable_weights:
                #         tf.summary.histogram(w.values[0].name, w, step=batch * BATCH_SIZE_PER_REPLICA)

    class SpeedCallback(tf.keras.callbacks.Callback):
        last_time_ck = None
        now_time_ck = None
        time_per_step = 0
        TIME_CK_PERIOD = 100

        def on_batch_end(self, batch, logs=None):
            # print(logs)
            if (batch // BATCH_SIZE) % self.TIME_CK_PERIOD == 0:
                now_time_ck = datetime.now()
                if self.last_time_ck:
                    time_per_step = (now_time_ck - self.last_time_ck) / self.TIME_CK_PERIOD
                    ms = (time_per_step.days * 86400000) + (time_per_step.seconds * 1000) + (
                            time_per_step.microseconds / 1000)
                    with train_summary_writer.as_default():
                        with tf.name_scope("training statistics"):
                            tf.summary.scalar('time_per_batch_ms', ms, step=(batch // BATCH_SIZE))

                    print(f"time per step {ms}ms")
                    remaining_batches = (DATASET_SIZE - batch) // BATCH_SIZE
                    hours = int((ms * remaining_batches) / 3600000)
                    minutes = int((ms * remaining_batches) / 60000) - hours * 60
                    print(f"ETA: {hours}:{minutes}")
                self.last_time_ck = now_time_ck

    class CodeLongformerTorch(torch.nn.Module):

        def __init__(self, voocab_size):
            super(CodeLongformerTorch, self).__init__()
            # Transformers CodeBERT model

            configuration = LongformerConfig(vocab_size=voocab_size, attention_window=ATTENTION_WINDOW,
                                             max_position_embeddings=1024 + 2)
            configuration.return_dict = False
            self._model = LongformerModel(configuration)

            # configuration = BertConfig(vocab_size=voocab_size)
            # self._model = BertModel(configuration)

            self.head_qa_sp_1 = torch.nn.Linear(configuration.hidden_size, 1000)
            self.relu = torch.nn.ReLU()
            self.head_qa_sp_2 = torch.nn.Linear(1000, 2)
            self.head_vocab = torch.nn.Linear(configuration.hidden_size, voocab_size)

        def forward(self, x):
            inputs, att_mask_vec, glob_loc_mask, token_type_vec = x

            toks, cls_hidden_1 = self._model(inputs, att_mask_vec, glob_loc_mask, token_type_ids=token_type_vec)
            toks_pbs = self.head_vocab(toks)
            cls_hidden_2 = self.relu(self.head_qa_sp_1(cls_hidden_1))
            qa_st = self.head_qa_sp_2(cls_hidden_2)
            return toks_pbs, qa_st

        def get_longformer(self):
            return self._model

    print(f"{my_id}: Creating model ...")

    files = [str(x) for x in Path(CKPT_DIR).glob(f"*.ckpt")]
    # print(f"{my_id}: ckpts : ", files)
    if len(files) > 0:
        file = sorted(files)[-1]
        checkpoint = torch.load(os.path.join(".", file), map_location=dev)
        print(f"{my_id}: Restored from {file}")
    else:
        checkpoint = None
        print("Initializing from scratch.")

    net = CodeLongformerTorch(VOCAB_SIZE)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = torch.nn.DataParallel(net)
    net.cuda(dev)
    if my_id == 0:
        wandb.watch(net)
    optimizer = torch.optim.Adam(net.parameters(), lr=INIT_LR)

    if not os.path.isdir(CKPT_DIR):
        os.mkdir(CKPT_DIR)

    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        net.load_state_dict(checkpoint['model_state_dict'])

    # export_for_publishing(train_dataset, net)

    lrs = LRScheduler()

    mlm_cross_entropy = torch.nn.CrossEntropyLoss()
    binary_crosentropy = torch.nn.BCEWithLogitsLoss(reduction="none")

    def qa_sp_cost(y_true, y_pred, mask):

        loss = binary_crosentropy(y_pred, y_true.type(torch.float32))
        loss_masked = torch.mul(loss, mask).type(torch.float32)

        return loss_masked

    callbacks = []
    if my_id == 0:
        print("Adding callbacks to master-0")
        callbacks = [
            SpeedCallback(),
            TBCallback(),
            SaveCkpt()
        ]
    callbacks.append(lrs)

    for batch, (examples, labels) in enumerate(train_dataset):
        example = [torch.tensor(e.numpy(), dtype=torch.int64).cuda(dev) for e in examples]
        outputs, qa_sp_lables, task_mask = [torch.tensor(e.numpy(), dtype=torch.int64).cuda(dev) for e in labels]
        optimizer.zero_grad()
        probs = net(example)

        preds = probs[0].reshape((BATCH_SIZE_PER_REPLICA * SEQ_LEN, VOCAB_SIZE))
        outs = outputs.reshape(BATCH_SIZE_PER_REPLICA * SEQ_LEN)
        loss_value1 = mlm_cross_entropy(preds, outs)

        loss_value2_log = torch.mean(qa_sp_cost(qa_sp_lables, probs[1], task_mask), dim=0)
        loss_value2 = torch.sum(loss_value2_log)
        loss_value = loss_value1 + loss_value2

        loss_value.backward()
        optimizer.step()

        a = loss_value2_log.cpu().detach()
        b = loss_value1.cpu().detach()
        c = loss_value2.cpu().detach()
        d = loss_value.cpu().detach()

        # print(loss_value)

        st, qa = a.numpy()[0], a.numpy()[1]
        logs = {"loss_sp": st, "loss_qa": qa, "loss": d.numpy(), "output_1_loss":
            b.cpu().detach().numpy(),
                "output_2_loss":
                    c.numpy(),
                "step": f"{(batch * BATCH_SIZE + start_from) // BATCH_SIZE}/{DATASET_SIZE / BATCH_SIZE}"}
        # print(logs)
        for callback in callbacks:
            callback.on_batch_end(batch * BATCH_SIZE + start_from, logs)


@tf.function
def build_masked_lm_data(input_ids, mask_id, predict_prob, mask_predicted_prob, perm_prob, num_tokens):
    output = tf.identity(input_ids)
    predict_mask = tf.squeeze(
        tf.random.categorical(tf.math.log([[1 - predict_prob, predict_prob]]), tf.shape(input_ids)[0]))
    output = output * predict_mask - 100 * (1 - predict_mask)
    mask_mask = tf.squeeze(
        tf.random.categorical(tf.math.log([[1 - mask_predicted_prob, mask_predicted_prob]]), tf.shape(input_ids)[0]))
    input_ids = input_ids * (1 - predict_mask * mask_mask) + mask_id * predict_mask * mask_mask
    replace_mask = tf.squeeze(tf.random.categorical(tf.math.log([[1 - perm_prob, perm_prob]]), tf.shape(input_ids)[0]))
    input_ids = input_ids * (1 - predict_mask * (1 - mask_mask) * replace_mask) + tf.random.uniform(tf.shape(input_ids),
                                                                                                    maxval=num_tokens,
                                                                                                    dtype=tf.int64) * predict_mask * (
        1 - mask_mask) * replace_mask
    return input_ids, output


if __name__ == '__main__':
    train()
