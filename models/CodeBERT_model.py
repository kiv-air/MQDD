import torch
from transformers import AutoConfig, AutoModel

FIRST_DROPOUT = 0.4
SECOND_DROPOUT = 0.3
HIDDEN_DROPOUT = 0.3
ATTENTION_DROPOUT = 0.3


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




