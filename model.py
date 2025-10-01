import torch 
import torch.nn as nn
from transformers import DistilBertModel

class SpamClassifier(nn.Module):
    def __init__(self, bert_model, num_features=5, dropout_rate=0.1):
        super().__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(bert_model.config.hidden_size + num_features, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask, features):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = bert_output.last_hidden_state[:, 0]
        combined = torch.cat((self.dropout(pooled_output), features), dim=1)
        return self.softmax(self.fc(combined))