import torch
import torch.nn as nn
from transformers import DistilBertTokenizer, DistilBertModel
from torchvision import models
import os

class DistilBertMultiLabel(nn.Module):
    def __init__(self, n_classes=3):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]
        pooled = self.dropout(pooled)
        return torch.sigmoid(self.classifier(pooled))


def load_category_model(model_path="models/category_model.pt", device="cpu"):
    model = DistilBertMultiLabel(n_classes=3)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def load_subcategory_model(model_path="models/subcategory_model.pth", device="cpu"):
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 11)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def load_tokenizer():
    return DistilBertTokenizer.from_pretrained('distilbert-base-uncased')