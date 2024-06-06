import random

import torch
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel


def extract_entity(sequence_output, e_mask):
    extended_e_mask = e_mask.unsqueeze(-1)
    extended_e_mask = extended_e_mask.float() * sequence_output
    extended_e_mask, _ = extended_e_mask.max(dim=-2)
    return extended_e_mask.float()


class PCL(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.relation_emb_dim = config.hidden_size
        self.temperature = config.temperature
        self.bert = BertModel(config)
        self.classifier = nn.Sequential(nn.Linear(self.relation_emb_dim, self.relation_emb_dim),
                                        nn.Tanh(),
                                        nn.Linear(self.relation_emb_dim, self.num_labels)
                                        )
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            marked_head=None,
            marked_tail=None,
            mask_mask=None,
            input_relation_emb=None,
            train_y_attr=None,
            train_y_e1=None,
            train_y_e2=None,
            labels=None
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
        )

        sequence_output = outputs[0]  # Sequence of hidden-states of the last layer
        e1_h = extract_entity(sequence_output, marked_head)
        e2_h = extract_entity(sequence_output, marked_tail)
        mask_mark = extract_entity(sequence_output, mask_mask)
        sentence_embeddings = torch.tanh(mask_mark)
        e1_h = torch.tanh(e1_h)
        e2_h = torch.tanh(e2_h)
        outputs = (outputs,)
        if labels is not None:
            ce_logit = self.classifier(sentence_embeddings)
            ce_loss = self.ce_loss(ce_logit.view(-1, self.num_labels), labels.view(-1))

            n = labels.shape[0]
            match_mask = torch.nonzero(
                torch.all(torch.eq(input_relation_emb.unsqueeze(1), train_y_attr.unsqueeze(0)), dim=2))[:, 1]
            rel_loss = cal_loss(train_y_attr, sentence_embeddings, labels)
            head_loss = cal_loss(train_y_e1, e1_h, labels)
            tail_loss = cal_loss(train_y_e2, e2_h, labels)

            outputs = (ce_loss + rel_loss + head_loss + tail_loss,)
        return outputs, sentence_embeddings, e1_h, e2_h
