from torch import nn
import torch
import torch.nn.functional as F

from logger import config_logger

# torch.set_printoptions(profile="full")

logger = config_logger("MODEL", "INFO")

class RoBERTaCoQA(nn.Module):
    """BERT model for CoQA Dataset.

    """

    def __init__(self, roberta):
        super(RoBERTaCoQA, self).__init__()
        self.hidden_size = 1024
        self.roberta = roberta
        self.INF = -1e4
        self._make_layers()
        self.loss = nn.CrossEntropyLoss()

    def _make_layers(self):
        self.logits_layer = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 2))
        self.yesnounk_layer = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 3))
        self.logits_layer.apply(self.init_weights)
        self.yesnounk_layer.apply(self.init_weights)

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, batch, answer_verification=False):
        input_ids, input_mask, context_mask, start_position, end_position, unk_label, \
        yes_label, no_label, extractive_label, _, _ = batch
        sequence_output = self.roberta.extract_features(input_ids)
        pooled_output = sequence_output[:, 0, :].squeeze(1)

        logits = self.logits_layer(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        logits = self.yesnounk_layer(pooled_output)
        yes_logits, no_logits, unk_logits = logits.split(1, dim=-1)

        batch_size = start_position.size(0)
        max_seq_len = input_ids.size(1)

        all_start_label = extractive_label * start_position + (1 - extractive_label) * (max_seq_len + no_label + 2*unk_label)
        all_end_label = extractive_label * end_position + (1 - extractive_label) * (max_seq_len + no_label + 2*unk_label)
        
        start_logits_ = start_logits * input_mask + (1 - input_mask) * self.INF
        end_logits_ = end_logits * input_mask + (1 - input_mask) * self.INF
        all_start_logits = torch.cat((start_logits_, yes_logits, no_logits, unk_logits), dim=-1)
        all_end_logits = torch.cat((end_logits_, yes_logits, no_logits, unk_logits), dim=-1)

        start_loss = self.loss(all_start_logits, all_start_label)
        end_loss = self.loss(all_end_logits, all_end_label)

        loss = (start_loss + end_loss) / 2.0
        output = {"start_logits": start_logits, "end_logits": end_logits, "yes_logits": yes_logits, \
                  "no_logits": no_logits, "unk_logits": unk_logits}
        return loss, output

