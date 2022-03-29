import torch
from transformers import LongformerConfig, LongformerModel

FIRST_DROPOUT = 0.4
SECOND_DROPOUT = 0.3
HIDDEN_DROPOUT = 0.3
ATTENTION_DROPOUT = 0.3


class ClsHeadModelMQDD(torch.nn.Module):

    def __init__(self, model_path):
        super(ClsHeadModelMQDD, self).__init__()
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


