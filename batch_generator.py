import torch
from torch.utils.data import TensorDataset, DataLoader,RandomSampler,SequentialSampler



class BatchGenerator(object):
    def __init__(self, data):
        all_input_ids = torch.tensor([i["input_ids"] for i in data], dtype=torch.long)
        all_input_mask = torch.tensor([i["input_mask"] for i in data], dtype=torch.float)
        all_context_mask = torch.tensor([i["context_mask"] for i in data], dtype=torch.float)
        all_start_positions = torch.tensor([i["start_position"] for i in data], dtype=torch.long)
        all_end_positions = torch.tensor([i["end_position"] for i in data], dtype=torch.long)
        all_unk_flag = torch.tensor([i["unk_flag"] for i in data], dtype=torch.long)
        all_yes_flag = torch.tensor([i["yes_flag"] for i in data], dtype=torch.long)
        all_no_flag = torch.tensor([i["no_flag"] for i in data], dtype=torch.long)
        all_extractive_flag = torch.tensor([i["extractive_flag"] for i in data], dtype=torch.long)
        all_rationale_mask = torch.tensor([i["rationale_mask"] for i in data], dtype=torch.float)
        all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)

        self.dataset = TensorDataset(all_input_ids, all_input_mask, all_context_mask, all_start_positions, 
                                     all_end_positions, all_unk_flag, all_yes_flag, all_no_flag, 
                                     all_extractive_flag, all_rationale_mask, all_example_index)


    def __call__(self, batch_size, is_train=True, num_workers=0, drop_last=False):
        self.batch_size = batch_size
        if is_train:
            self.sampler = RandomSampler(self.dataset)
        else:
            self.sampler = SequentialSampler(self.dataset)
        self.num_workers = num_workers

        return DataLoader(dataset=self.dataset, batch_size=self.batch_size, sampler=self.sampler, 
                          num_workers=self.num_workers, drop_last=drop_last)
