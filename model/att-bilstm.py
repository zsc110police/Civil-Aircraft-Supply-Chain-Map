#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Version : Python 3.6

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import BertModel, BertConfig


class BiLSTM(nn.Module):
    def __init__(self, class_num, config):
        super().__init__()
        self.class_num = class_num

        # hyper parameters
        self.max_len = config.max_len
        self.hidden_size = config.hidden_size
        self.layers_num = config.layers_num
        self.lstm_dropout_value = config.lstm_dropout
        self.linear_dropout_value = config.linear_dropout

        # BERT encoder
        bert_config = BertConfig.from_pretrained(config.embedding_path)
        self.bert = BertModel.from_pretrained(config.embedding_path, config=bert_config)
        self.word_dim = bert_config.hidden_size

        # BiLSTM
        self.lstm = nn.LSTM(
            input_size=self.word_dim,
            hidden_size=self.hidden_size,
            num_layers=self.layers_num,
            batch_first=True,
            dropout=0.1,
            bidirectional=True,
        )

        self.lstm_dropout = nn.Dropout(self.lstm_dropout_value)
        self.linear_dropout = nn.Dropout(self.linear_dropout_value)

        # Dense layer (Classifier)
        self.dense = nn.Linear(
            in_features=self.hidden_size * 2,  # Bi-LSTM
            out_features=self.class_num,
            bias=True
        )

        # 初始化
        nn.init.xavier_normal_(self.dense.weight)
        nn.init.constant_(self.dense.bias, 0.)

    def lstm_layer(self, x, mask):
        lengths = torch.sum(mask.gt(0), dim=-1)
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        output, _ = self.lstm(packed)
        output, _ = pad_packed_sequence(output, batch_first=True, padding_value=0.0, total_length=self.max_len)
        return output  # [B, L, H*2]

    def forward(self, input_ids, attention_mask):
        # 1. BERT embedding
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        emb = outputs.last_hidden_state  # [B, L, H]

        # 2. BiLSTM
        lstm_out = self.lstm_layer(emb, attention_mask)
        lstm_out = self.lstm_dropout(lstm_out)

        # 3. Obtain the sentence representation by using the last time step
        lengths = torch.sum(attention_mask > 0, dim=1) - 1  # [B]
        idx = lengths.view(-1, 1, 1).expand(-1, 1, lstm_out.size(2))  # [B,1,H*2]
        reps = lstm_out.gather(1, idx).squeeze(1)  # [B, H*2]

        reps = self.linear_dropout(reps)

        # 4. 分类输出
        logits = self.dense(reps)  # [B, class_num]
        return logits
