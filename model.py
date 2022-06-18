from torch import nn
import torch
from logger import config_logger
from config import args
# torch.set_printoptions(profile="full")

logger = config_logger("MODEL", "INFO")

class RoBERTaCoQA(nn.Module):
    """RoBERTa model for CoQA Dataset.

    """
    def __init__(self, roberta):
        super(RoBERTaCoQA, self).__init__()
        self.hidden_size = 1024
        self.roberta = roberta
        self.INF = -1e4
        self._make_layers()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)
        self.logSoftmax = nn.LogSoftmax(dim=-1)
        self.loss = nn.CrossEntropyLoss()
        self.bce = nn.BCELoss()
        self.kl = nn.KLDivLoss(reduction='batchmean')
        if args.vat_loss:
            self.logits_backup = {}
    def backup_logits(self, start_logits, end_logits, rationale_logits, _id):
        self.logits_backup[_id] = {}
        self.logits_backup[_id]['start_logits'] = start_logits.detach()
        self.logits_backup[_id]['end_logits'] = end_logits.detach()
        self.logits_backup[_id]['rationale_logits'] = rationale_logits.detach()
        
    def clear_backup(self):
        self.logits_backup = {}
    def _make_layers(self):
        self.rationale_layer = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1))
        self.attention_layer = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1))
        self.logits_layer = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 2))
        self.yesnounk_layer = nn.Sequential(
            nn.Linear(2*self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 3))
        self.rationale_layer.apply(self.init_weights)
        self.attention_layer.apply(self.init_weights)
        self.logits_layer.apply(self.init_weights)
        self.yesnounk_layer.apply(self.init_weights)

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
    def em_loss(self, x):
        return  -(self.softmax(x) * self.logSoftmax(x)).sum(dim=-1).mean()
    def forward(self, batch, answer_verification=True, loss_func='bce', optim_at = False):
        input_ids, input_mask, context_mask, start_position, end_position, unk_label, \
        yes_label, no_label, extractive_label, rationale_mask, _ = batch
        _id = start_position
        sequence_output = self.roberta.extract_features(input_ids)
        pooled_output = sequence_output[:, 0, :].squeeze(1)
        rationale_logits = self.sigmoid(self.rationale_layer(sequence_output).squeeze(-1))
        rationale_sequence_output = rationale_logits.unsqueeze(-1) * sequence_output
        
        logits = self.logits_layer(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        attention_logits = self.softmax(input_mask * self.attention_layer(rationale_sequence_output).squeeze(-1) + (1-input_mask) * self.INF)
        sum_att_output = torch.sum(attention_logits.unsqueeze(-1) * sequence_output, dim=1)
        logits = self.yesnounk_layer(torch.cat([pooled_output, sum_att_output], dim=1))
        yes_logits, no_logits, unk_logits = logits.split(1, dim=-1)

        max_seq_len = input_ids.size(1)

        all_start_label = extractive_label * start_position + (1 - extractive_label) * (max_seq_len + no_label + 2*unk_label)
        all_end_label = extractive_label * end_position + (1 - extractive_label) * (max_seq_len + no_label + 2*unk_label)
        
        start_logits_ = start_logits * input_mask + (1 - input_mask) * self.INF
        end_logits_ = end_logits * input_mask + (1 - input_mask) * self.INF
        all_start_logits = torch.cat((start_logits_, yes_logits, no_logits, unk_logits), dim=-1)
        all_end_logits = torch.cat((end_logits_, yes_logits, no_logits, unk_logits), dim=-1)
        output = {"start_logits": start_logits_, "end_logits": end_logits_, "yes_logits": yes_logits, \
                "no_logits": no_logits, "unk_logits": unk_logits}
        if args.do_predict:
            return None, output
        if loss_func == 'bce':
            if (not optim_at) and args.vat_loss: # backup logits for KLDiv in VAT
                self.backup_logits(all_start_logits, all_end_logits, rationale_logits, _id)
            start_loss = self.loss(all_start_logits, all_start_label)
            end_loss = self.loss(all_end_logits, all_end_label)
            #logger.info(rationale_logits)
            #logger.info(rationale_mask)
            rationale_loss = self.bce(rationale_logits, rationale_mask)
            loss = (start_loss + end_loss) / 2.0 + 5 * rationale_loss
            if args.em_loss and not optim_at:
                # em loss
                em_start_loss = self.em_loss(all_start_logits)
                em_end_loss = self.em_loss(all_end_logits)
                em_rationale_loss = self.em_loss(rationale_logits)
                # print('loss:')
                # print(loss)
                # print('em loss:')
                # print(args.l_em * (em_start_loss + em_end_loss + em_rationale_loss) / 3.0)
                loss += args.l_em * (em_start_loss + em_end_loss + em_rationale_loss) / 3.0
        if loss_func == 'kl':
            start_loss = self.kl( self.logSoftmax(self.logits_backup[_id]['start_logits']), self.softmax(all_start_logits))
            end_loss = self.kl(self.logSoftmax(self.logits_backup[_id]['end_logits']), self.softmax(all_end_logits))
            rationale_loss = self.kl( self.logSoftmax(self.logits_backup[_id]['rationale_logits']), self.softmax(rationale_logits))
            loss = (start_loss + end_loss + rationale_loss) / 3.0
            
        
        return loss, output

