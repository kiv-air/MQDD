import os
import time

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import torch
import wandb
from transformers import AutoTokenizer, BertTokenizerFast, AutoConfig, AutoModel, LongformerConfig, LongformerModel
from sklearn.metrics import f1_score

if os.name != 'nt':
    BUFFER_SIZE = 200
else:
    BUFFER_SIZE = 10

LABEL_MAPPING = {0: 1,
                 1: 0,
                 2: 0,
                 3: 0}

DATASET_SIZE = 419200

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device {device}")


class CosineSimilarityModelCodeBERT(torch.nn.Module):

    def __init__(self):
        super(CosineSimilarityModelCodeBERT, self).__init__()
        # Transformers CodeBERT model
        self._model_config = AutoConfig.from_pretrained("microsoft/codebert-base",
                                                        hidden_dropout_prob=HIDDEN_DROPOUT,
                                                        attention_probs_dropout_prob=ATTENTION_DROPOUT
                                                        )
        self._model_config.return_dict = False
        self.bert_model = AutoModel.from_pretrained("microsoft/codebert-base", config=self._model_config)
        self._cos = torch.nn.CosineSimilarity()

    def forward(self, x):
        input_1, input_1_mask, input_2, input_2_mask = torch.split(x, 1, dim=1)
        input_1, input_1_mask = torch.squeeze(input_1), torch.squeeze(input_1_mask)
        input_2, input_2_mask = torch.squeeze(input_2), torch.squeeze(input_2_mask)
        input_1_dense = self.bert_model(input_1, input_1_mask)[1]
        input_2_dense = self.bert_model(input_2, input_2_mask)[1]
        predictions = (self._cos(input_1_dense, input_2_dense) + 1) / 2
        predictions_0_cls = 1 - predictions
        predictions = torch.stack((predictions_0_cls, predictions), 1)
        return predictions

    def get_L2_loss(self):
        return torch.tensor(0, dtype=torch.float32)


class ClsHeadModelCodeBERT(torch.nn.Module):

    def __init__(self):
        super(ClsHeadModelCodeBERT, self).__init__()
        # Transformers CodeBERT model
        self._model_config = AutoConfig.from_pretrained("microsoft/codebert-base",
                                                        hidden_dropout_prob=HIDDEN_DROPOUT,
                                                        attention_probs_dropout_prob=ATTENTION_DROPOUT)
        self._model_config.return_dict = False
        self.bert_model = AutoModel.from_pretrained("microsoft/codebert-base", config=self._model_config)
        self._dropout1 = torch.nn.Dropout(FIRST_DROPOUT)
        self._dropout2 = torch.nn.Dropout(SECOND_DROPOUT)
        self._dense = torch.nn.Linear(1536, 256, bias=True)
        self._out = torch.nn.Linear(256, 2, bias=False)
        self._relu = torch.nn.ReLU()
        self._softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        input_1, input_1_mask, input_2, input_2_mask = torch.split(x, 1, dim=1)
        input_1, input_1_mask = torch.squeeze(input_1), torch.squeeze(input_1_mask)
        input_2, input_2_mask = torch.squeeze(input_2), torch.squeeze(input_2_mask)
        input_1_dense = self.bert_model(input_1, input_1_mask)[1]
        input_2_dense = self.bert_model(input_2, input_2_mask)[1]
        concat = torch.cat((input_1_dense, input_2_dense), 1)
        x = self._dropout1(concat)
        x = self._relu(self._dense(x))
        x = self._dropout2(x)
        return self._out(x)
        # return self._softmax(self._out(x))

    def get_L2_loss(self):
        return torch.tensor(0, dtype=torch.float32)


class CosineSimilarityModelFERDA(torch.nn.Module):

    def __init__(self, model_path):
        super(CosineSimilarityModelFERDA, self).__init__()
        # Transformers Longformer model
        self._model_config = LongformerConfig.from_pretrained(model_path,
                                                              hidden_dropout_prob=HIDDEN_DROPOUT,
                                                              attention_probs_dropout_prob=ATTENTION_DROPOUT)
        self._model_config.return_dict = False
        self.bert_model = LongformerModel.from_pretrained(model_path, config=self._model_config)

        self._cos = torch.nn.CosineSimilarity()

    def forward(self, x):
        input_1, input_1_mask, input_1_tok_types, input_2, input_2_mask, input_2_tok_types = torch.split(x, 1, dim=1)
        input_1, input_1_mask, input_1_tok_types = torch.squeeze(input_1), torch.squeeze(input_1_mask), torch.squeeze(
            input_1_tok_types)
        input_2, input_2_mask, input_2_tok_types = torch.squeeze(input_2), torch.squeeze(input_2_mask), torch.squeeze(
            input_2_tok_types)
        input_1_dense = self.bert_model(input_1, input_1_mask, None, token_type_ids=input_1_tok_types)[1]
        input_2_dense = self.bert_model(input_2, input_2_mask, None, token_type_ids=input_2_tok_types)[1]
        predictions = (self._cos(input_1_dense, input_2_dense) + 1) / 2
        predictions_0_cls = 1 - predictions
        predictions = torch.stack((predictions_0_cls, predictions), 1)
        return predictions

    def get_L2_loss(self):
        return torch.tensor(0, dtype=torch.float32)


class ClsHeadModelFERDA(torch.nn.Module):

    def __init__(self, model_path):
        super(ClsHeadModelFERDA, self).__init__()
        # Transformers Longformer model
        self._model_config = LongformerConfig.from_pretrained(model_path,
                                                              hidden_dropout_prob=HIDDEN_DROPOUT,
                                                              attention_probs_dropout_prob=ATTENTION_DROPOUT)
        self._model_config.return_dict = False
        self.bert_model = LongformerModel.from_pretrained(model_path, config=self._model_config)
        self._dropout1 = torch.nn.Dropout(FIRST_DROPOUT)
        self._dropout2 = torch.nn.Dropout(SECOND_DROPOUT)
        self._dense = torch.nn.Linear(1536, 256, bias=True)
        self._out = torch.nn.Linear(256, 2, bias=False)
        self._relu = torch.nn.ReLU()
        self._softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        input_1, input_1_mask, input_1_tok_types, input_2, input_2_mask, input_2_tok_types = torch.split(x, 1, dim=1)
        input_1, input_1_mask, input_1_tok_types = torch.squeeze(input_1), torch.squeeze(input_1_mask), torch.squeeze(
            input_1_tok_types)
        input_2, input_2_mask, input_2_tok_types = torch.squeeze(input_2), torch.squeeze(input_2_mask), torch.squeeze(
            input_2_tok_types)
        input_1_dense = self.bert_model(input_1, input_1_mask, None, token_type_ids=input_1_tok_types)[1]
        input_2_dense = self.bert_model(input_2, input_2_mask, None, token_type_ids=input_2_tok_types)[1]

        concat = torch.cat((input_1_dense, input_2_dense), 1)
        x = self._dropout1(concat)
        x = self._relu(self._dense(x))
        x = self._dropout2(x)
        return self._out(x)
        # return self._softmax(self._out(x))

    def get_L2_loss(self):
        return torch.tensor(0, dtype=torch.float32)


def encode_codebert(x, label):
    codebert_tokenizer = AutoTokenizer.from_pretrained("./code-bert-base", use_fast=True)
    sentence1, sentence2, code1, code2 = x
    sentence1, sentence2, code1, code2 = sentence1.numpy(), sentence2.numpy(), code1.numpy(), code2.numpy()

    def process_one(sentence, code):
        sentence = codebert_tokenizer.tokenize(f"{sentence.decode('utf-8')}")
        code = codebert_tokenizer.tokenize(f"{code.decode('utf-8')}")

        sentence = codebert_tokenizer.convert_tokens_to_ids(sentence)
        code = codebert_tokenizer.convert_tokens_to_ids(code)

        return codebert_tokenizer.prepare_for_model(sentence,
                                                    code,
                                                    truncation="longest_first",
                                                    padding="max_length",
                                                    max_length=MAX_SEQ_LEN,
                                                    return_tensors="np")

    # preprocess the input for BERT
    input1 = process_one(sentence1, code1)
    input2 = process_one(sentence2, code2)

    label = LABEL_MAPPING[label.numpy()]

    res = (input1["input_ids"], input1["attention_mask"],
           input2["input_ids"], input2["attention_mask"])
    return res, label


def encode_map_function_codebert(x, label):
    x, label = tf.py_function(encode_codebert, inp=[x, label], Tout=(tf.int32, tf.int64))

    x.set_shape([4, MAX_SEQ_LEN])
    label.set_shape([])

    return x, label


def encode_ferda(x, label):
    ferda_tokenizer = BertTokenizerFast.from_pretrained(TOKENIZER_PATH, strip_accents=False)
    sentence1, sentence2, code1, code2 = x
    sentence1, sentence2, code1, code2 = sentence1.numpy(), sentence2.numpy(), code1.numpy(), code2.numpy()

    def process_one(sentence, code):
        sentence = ferda_tokenizer.tokenize(f"{sentence.decode('utf-8')}")
        code = ferda_tokenizer.tokenize(f"{code.decode('utf-8')}")

        sentence = ferda_tokenizer.convert_tokens_to_ids(sentence)
        code = ferda_tokenizer.convert_tokens_to_ids(code)

        return ferda_tokenizer.prepare_for_model(sentence,
                                                 code,
                                                 truncation="longest_first",
                                                 padding="max_length",
                                                 max_length=MAX_SEQ_LEN,
                                                 return_tensors="np",
                                                 return_token_type_ids=True)

    # preprocess the input for BERT
    input1 = process_one(sentence1, code1)
    input2 = process_one(sentence2, code2)

    label = LABEL_MAPPING[label.numpy()]

    if len(input1["token_type_ids"]) > MAX_SEQ_LEN:
        input1["token_type_ids"] = input1["token_type_ids"][:MAX_SEQ_LEN]

    if len(input2["token_type_ids"]) > MAX_SEQ_LEN:
        input2["token_type_ids"] = input2["token_type_ids"][:MAX_SEQ_LEN]

    # glob_loc_att_mask = np.concatenate([np.ones(1, dtype=np.int32), np.zeros(MAX_SEQ_LEN-1, dtype=np.int32)])
    attention_mask1 = input1["attention_mask"]
    attention_mask2 = input2["attention_mask"]

    attention_mask1[0] = 2
    attention_mask2[0] = 2

    res = (input1["input_ids"], attention_mask1, input1["token_type_ids"],
           input2["input_ids"], attention_mask2, input2["token_type_ids"])
    return res, label


def encode_map_function_ferda(x, label):
    x, label = tf.py_function(encode_ferda, inp=[x, label], Tout=(tf.int32, tf.int64))

    x.set_shape([6, MAX_SEQ_LEN])
    label.set_shape([])

    return x, label


def prepare_dataset(data_dir, batch_size, encode_map_fn, skip=0):
    np.random.seed(SEED)

    duplicates, info = tfds.load("stackexchange/stackoverflow_preprocessed_duplicates", with_info=True,
                                 data_dir=data_dir)
    fulltext_similarities = tfds.load("stackexchange/stackoverflow_preprocessed_fulltext_similarities", with_info=False,
                                      data_dir=data_dir)
    tag_similarities = tfds.load("stackexchange/stackoverflow_preprocessed_tag_similarities", with_info=False,
                                 data_dir=data_dir)
    different = tfds.load("stackexchange/stackoverflow_preprocessed_different", with_info=False, data_dir=data_dir)

    train_part_size = info.splits["train"].num_examples
    validation_part_size = info.splits["validation"].num_examples

    train_choice = np.concatenate([
        np.full((train_part_size,), 0, dtype=np.int64),
        np.full((train_part_size // 3,), 1, dtype=np.int64),
        np.full((train_part_size // 3,), 2, dtype=np.int64),
        np.full((train_part_size // 3,), 3, dtype=np.int64),
    ])
    np.random.shuffle(train_choice)

    validation_choice = np.concatenate([
        np.full((validation_part_size,), 0, dtype=np.int64),
        np.full((validation_part_size // 3,), 1, dtype=np.int64),
        np.full((validation_part_size // 3,), 2, dtype=np.int64),
        np.full((validation_part_size // 3,), 3, dtype=np.int64),
    ])
    np.random.shuffle(validation_choice)

    train_ds = tf.data.experimental.choose_from_datasets([duplicates["train"],
                                                          fulltext_similarities["train"],
                                                          tag_similarities["train"],
                                                          different["train"]],
                                                         tf.data.Dataset.from_tensor_slices(train_choice))

    validation_ds = tf.data.experimental.choose_from_datasets([duplicates["validation"],
                                                               fulltext_similarities["validation"],
                                                               tag_similarities["validation"],
                                                               different["validation"]],
                                                              tf.data.Dataset.from_tensor_slices(
                                                                  validation_choice))

    def sentence_pair_map_fn(e):
        sentence1 = e["first_question_text"]
        sentence2 = e["second_question_text"]
        code1 = e["first_question_code"]
        code2 = e["second_question_code"]
        label = e["label"]

        return (sentence1, sentence2, code1, code2), label

    # skip
    print(f"Skipping {skip} batches, {skip * BATCH_SIZE} examples")
    train_ds = train_ds.skip(skip * BATCH_SIZE)
    train_ds = train_ds.map(sentence_pair_map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    validation_ds = validation_ds.map(sentence_pair_map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    train_ds = train_ds.map(encode_map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    validation_ds = validation_ds.map(encode_map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    train_ds = train_ds.shuffle(BUFFER_SIZE, reshuffle_each_iteration=False)
    validation_ds = validation_ds.shuffle(BUFFER_SIZE, reshuffle_each_iteration=False)

    train_ds = train_ds.batch(batch_size, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
    validation_ds = validation_ds.batch(batch_size, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)

    return train_ds, validation_ds


class F1Loss(torch.nn.Module):
    '''Calculate F1 score. Can work with gpu tensors

    The original implementation is written by Michal Haltuf on Kaggle.

    Returns
    -------
    torch.Tensor
        `ndim` == 1. epsilon <= val <= 1

    Reference
    ---------
    - https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric
    - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
    - https://discuss.pytorch.org/t/calculating-precision-recall-and-f1-score-in-case-of-multi-label-classification/28265/6
    - http://www.ryanzhang.info/python/writing-your-own-loss-function-module-for-pytorch/
    '''

    def __init__(self, epsilon=1e-7):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, y_pred, y_true, ):
        import torch.nn.functional as F

        y_true = y_true.to(torch.int64)

        assert y_pred.ndim == 2
        assert y_true.ndim == 1

        y_true = F.one_hot(y_true, -1).to(torch.float32)
        y_pred = F.softmax(y_pred, dim=1)

        tp = (y_true * y_pred).sum(dim=0).to(torch.float32)
        tn = ((1 - y_true) * (1 - y_pred)).sum(dim=0).to(torch.float32)
        fp = ((1 - y_true) * y_pred).sum(dim=0).to(torch.float32)
        fn = (y_true * (1 - y_pred)).sum(dim=0).to(torch.float32)

        precision = tp / (tp + fp + self.epsilon)
        recall = tp / (tp + fn + self.epsilon)

        f1 = 2 * (precision * recall) / (precision + recall + self.epsilon)
        f1 = f1.clamp(min=self.epsilon, max=1 - self.epsilon)
        return 1 - f1.mean()


def optimizer_to(optimizer, dev):
    for param in optimizer.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(dev)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(dev)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(dev)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(dev)


def train(dataset_fn, net, optimizer, epochs=1, logging_steps=100, eval_steps=100,
          initial_step=0, initial_epoch=0):
    # Create loss function
    if LOSS == "F1":
        loss_fn = F1Loss().to(device)
    else:
        loss_fn = torch.nn.CrossEntropyLoss().to(device)

    runners_first_epoch = True
    step = initial_step

    for epoch in range(initial_epoch, epochs):
        running_loss = 0.0
        running_accuracy = 0.0
        running_f1_score = 0.0
        start_time = time.time()

        to_skip = 0
        if runners_first_epoch:
            to_skip = initial_step % STEPS_PER_EPOCH
            runners_first_epoch = False

        train_ds, validation_ds = dataset_fn(to_skip)

        for i, data in enumerate(train_ds, 0):
            step += 1
            inputs, labels = data
            inputs = torch.tensor(inputs.numpy(), dtype=torch.int64)
            if LOSS == "F1":
                labels = torch.tensor(labels.numpy(), dtype=torch.float)
            else:
                labels = torch.tensor(labels.numpy(), dtype=torch.int64)
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = net(inputs)
            l2_norm = 0
            for param in net.named_parameters():
                if "_out" in param[0] or "_dense" in param[0]:
                    l2_norm += torch.norm(param[1])
            loss = loss_fn(outputs, labels) + (l2_norm * L2_ALPHA)
            loss.backward()
            optimizer.step()

            outs = outputs.cpu().detach()
            lab = labels.cpu().detach()

            running_loss += loss.cpu().detach().item()
            binary_output = torch.argmax(outs, 1)
            # noinspection PyTypeChecker
            running_accuracy += torch.sum(binary_output == lab).item()
            running_f1_score += f1_score(lab, binary_output, labels=[0, 1], average="macro")
            del inputs
            del labels
            del outputs
            del loss
            del outs
            del binary_output
            del l2_norm

            if step % logging_steps == logging_steps - 1:
                total_time = time.time() - start_time
                start_time = time.time()

                resulting_loss = running_loss / logging_steps
                resulting_accuracy = running_accuracy / (logging_steps * len(lab))
                resulting_f1_score = running_f1_score / logging_steps
                running_loss = 0.0
                running_accuracy = 0.0
                running_f1_score = 0.0
                print('\r[epoch: %d, step: %5d] loss: %.3f accuracy: %.3f f1_score: %.3f %02d:%02d' % (
                    epoch + 1, step, resulting_loss, resulting_accuracy, resulting_f1_score, int(total_time / 60),
                    total_time % 60), end="")

                log_data = {"epoch": epoch + 1,
                            "lr": optimizer.state_dict()["param_groups"][0]["lr"],
                            "loss": resulting_loss,
                            "accuracy": resulting_accuracy,
                            "f1_score": resulting_f1_score}
                wandb.log(log_data, step=step)
                del lab
            else:
                del lab

            if step % SAVING_STEPS == SAVING_STEPS - 1:

                if CHECKPOINT is None:
                    ckpt_dir = os.path.join(CHECKPOINT_BASE_PATH, wandb.run.id)

                else:
                    ckpt_dir = os.path.join(CHECKPOINT_BASE_PATH, CHECKPOINT)

                if not os.path.exists(ckpt_dir):
                    os.mkdir(ckpt_dir)
                    if os.name != 'nt':
                        os.system(f"chgrp air {ckpt_dir}")
                        os.system(f"chmod ug+rwx {ckpt_dir}")
                ckpt_path = os.path.join(ckpt_dir, 'model.pt')
                if os.path.exists(ckpt_path):
                    os.remove(ckpt_path)
                    print(f"Saving ckpt, epoch: {epoch}, step: {step}, run_id: {wandb.run.id}")
                torch.save({
                    "epoch": epoch,
                    "step": step,
                    "model_state": model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "run_id": wandb.run.id
                }, ckpt_path)

                if os.name != 'nt':
                    os.system(f"chgrp air {ckpt_path}")
                    os.system(f"chmod ug+rwx {ckpt_path}")

            if step % eval_steps == eval_steps - 1:
                eval_running_loss = 0.0
                eval_running_accuracy = 0.0
                eval_running_f1_score = 0.0
                eval_batches = 0

                with torch.no_grad():
                    print("Running evaluation")
                    for j, eval_data in enumerate(validation_ds, 0):
                        eval_batches += 1
                        eval_inputs, eval_labels = eval_data
                        eval_inputs = torch.tensor(eval_inputs.numpy(), dtype=torch.int64)
                        if LOSS == "F1":
                            eval_labels = torch.tensor(eval_labels.numpy(), dtype=torch.float)
                        else:
                            eval_labels = torch.tensor(eval_labels.numpy(), dtype=torch.int64)

                        eval_inputs, eval_labels = eval_inputs.to(device), eval_labels.to(device)
                        eval_outputs = net(eval_inputs)
                        eval_running_loss += loss_fn(eval_outputs, eval_labels).item()
                        eval_binary_output = torch.argmax(eval_outputs, 1)
                        eval_running_accuracy += torch.sum(eval_binary_output == eval_labels)
                        eval_running_f1_score += f1_score(eval_labels.to("cpu"), eval_binary_output.to("cpu"),
                                                          labels=[0, 1], average="macro")

                    eval_loss = eval_running_loss / eval_steps
                    eval_accuracy = eval_running_accuracy / (eval_batches * len(eval_labels))
                    eval_f1_score = eval_running_f1_score / eval_batches
                    eval_log_data = {"val_loss": eval_loss,
                                     "val_accuracy": eval_accuracy,
                                     "val_f1_score": eval_f1_score}
                    wandb.log(eval_log_data, step=step)


if __name__ == '__main__':
    import json

    LR = 0.000005
    MAX_SEQ_LEN = 128
    BATCH_SIZE = 100
    LOSS = "CrossEntropy"
    SAVING_STEPS = 500
    LOGGING_STEPS = 100
    EVAL_STEPS = 4192
    EPOCHS = 10
    MODEL_PATH = "./model"
    TOKENIZER_PATH = "./model"
    DATASET_PATH = "./tensorflow_datasets"
    L2_ALPHA = 0.03
    FIRST_DROPOUT = 0.4
    SECOND_DROPOUT = 0.3
    HIDDEN_DROPOUT = 0.3
    ATTENTION_DROPOUT = 0.3
    CHECKPOINT_BASE_PATH = "/storage/plzen1/home/pasekj/code-bert-ckpts"
    CHECKPOINT = None
    MODEL_TYPE = "ClsFERDA"
    FREEZE_BASE = False
    SEED = np.random.randint(low=0, high=65535, dtype=np.uint32)

    config = None
    with open("model/config.json", "r") as f:
        config = json.load(f)

    ckpt = None
    ckpt_run_id = None
    if CHECKPOINT is not None:
        ckpt = torch.load(os.path.join(CHECKPOINT_BASE_PATH, CHECKPOINT, 'model.pt'), map_location="cpu")
        ckpt_run_id = ckpt["run_id"]

    config["lr"] = LR
    config["max_seq_len"] = MAX_SEQ_LEN
    config["batch_size"] = BATCH_SIZE
    config["loss_fn"] = LOSS
    config["model"] = MODEL_TYPE
    config["l2"] = L2_ALPHA
    config["first_dropout"] = FIRST_DROPOUT
    config["second_dropout"] = SECOND_DROPOUT
    config["hidden_dropout_prob"] = HIDDEN_DROPOUT
    config["attention_probs_dropout_prob"] = ATTENTION_DROPOUT
    config["freeze_base"] = FREEZE_BASE
    config["seed"] = SEED
    config["task"] = "so_duplicates"

    print("Initializing new run")
    wandb.init(
        project="code-bert-finetune",
        name=os.environ["WANDB_NEW_NAME"],
        save_code=True,
        reinit=True,
        resume="allow",
        config=config,
    )

    wandb.define_metric("val_loss", summary="min")
    wandb.define_metric("val_accuracy", summary="max")
    wandb.define_metric("val_f1_score", summary="max")

    LR = wandb.config["lr"]
    MAX_SEQ_LEN = wandb.config["max_seq_len"]
    BATCH_SIZE = wandb.config["batch_size"]
    LOSS = wandb.config["loss_fn"]
    MODEL_TYPE = wandb.config["model"]
    L2_ALPHA = wandb.config["l2"]
    FIRST_DROPOUT = wandb.config["first_dropout"]
    SECOND_DROPOUT = wandb.config["second_dropout"]
    SEED = wandb.config["seed"]
    FREEZE_BASE = config["freeze_base"]
    HIDDEN_DROPOUT = config["hidden_dropout_prob"]
    ATTENTION_DROPOUT = config["attention_probs_dropout_prob"] = ATTENTION_DROPOUT

    # print dropout setup
    print(f"Hidden dr: {HIDDEN_DROPOUT}")
    print(f"Attention dr: {ATTENTION_DROPOUT}")
    print(f"First dr: {FIRST_DROPOUT}")
    print(f"Second dr: {SECOND_DROPOUT}")

    STEPS_PER_EPOCH = DATASET_SIZE // BATCH_SIZE
    encode_fn = None

    # CREATE MODEL
    if MODEL_TYPE == "ClsFERDA":
        encode_fn = encode_map_function_ferda
        model = ClsHeadModelFERDA(MODEL_PATH)
    elif MODEL_TYPE == "CosFERDA":
        encode_fn = encode_map_function_ferda
        model = CosineSimilarityModelFERDA(MODEL_PATH)
    elif MODEL_TYPE == "CosCodeBERT":
        encode_fn = encode_map_function_codebert
        model = CosineSimilarityModelCodeBERT()
    elif MODEL_TYPE == "ClsCodeBERT":
        encode_fn = encode_map_function_codebert
        model = ClsHeadModelCodeBERT()
    else:
        model = None

    # LOAD CKPTS
    if ckpt is not None:
        print("Loading model checkpoint")
        model.load_state_dict(ckpt["model_state"])
        model.train()

    # freeze weights if necessary
    if FREEZE_BASE:
        print("Freezing weights of the pretrained model")
        for param in model.bert_model.parameters():
            param.requires_grad = False

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model)

    model.to(device)

    # CREATE OPTIMIZER
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    if ckpt is not None:
        print("Loading optimizer checkpoint")
        opt.load_state_dict(ckpt["optimizer_state_dict"])
        optimizer_to(opt, device)

    # SET INIT STEP AND INIT EPOCH -> then delete the checkpoint from the memory
    init_epoch = 0
    init_step = 0
    if ckpt is not None:
        init_epoch = ckpt["epoch"]
        init_step = ckpt["step"]
        del ckpt

    # FLUSH CACHE MEMORY
    torch.cuda.empty_cache()

    # TRAIN THE MODEL
    wandb.watch(model)
    train(lambda skip: prepare_dataset(DATASET_PATH, BATCH_SIZE, encode_fn, skip),
          model,
          opt,
          EPOCHS,
          LOGGING_STEPS,
          EVAL_STEPS,
          init_step,
          init_epoch)
