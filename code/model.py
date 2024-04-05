import torch
import torch.nn as nn
from transformers import AutoModel
from torch.cuda.amp import autocast
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        CE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-CE_loss)
        F_loss = self.alpha * (1 - pt)**self.gamma * CE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss

class CustomModel(nn.Module):
    def __init__(self, MODEL_NAME, config):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(MODEL_NAME, config=config)
        hidden_size = config.hidden_size
        self.loss_fnt = FocalLoss()
        self.fc_layer = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=0.1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 30)
        )

    @autocast()
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None, ss=None, os=None):
        outputs = self.encoder(
            input_ids,
            attention_mask=attention_mask, 
            token_type_ids=token_type_ids
        )
        pooled_output = outputs[0]
        cls_emb = pooled_output[:, 0]
        idx = torch.arange(input_ids.size(0)).to(input_ids.device)
        ss_emb = pooled_output[idx, ss]
        os_emb = pooled_output[idx, os]
        ent_emb = self.fc_layer(torch.cat((ss_emb, os_emb), dim=-1))
        h = self.fc_layer(torch.cat((ent_emb, cls_emb), dim=-1))
        logits = self.classifier(h)
        outputs = (logits,)

        if labels is not None:
            loss = self.loss_fnt(logits.float(), labels)
            outputs = (loss,) + outputs

        return outputs
    
