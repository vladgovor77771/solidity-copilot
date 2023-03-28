import os
import torch
from transformers import RobertaTokenizer, T5ForConditionalGeneration, TrainingArguments, Trainer
from datasets import Dataset
from torch.utils.data import Dataset as PTDataset
import logging

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

cache_path = './cache'
results_path = './results'
logs_path = './logs'

device = torch.device('cuda')

logging.getLogger(
    "transformers.tokenization_utils_base").setLevel(logging.ERROR)

class FunctionDataset(PTDataset):
    functions_ds: Dataset
    tokenizer: RobertaTokenizer

    def __init__(self):
        self.functions_ds = Dataset.load_from_disk('/data/functions_ds')
        self.tokenizer = RobertaTokenizer.from_pretrained('Salesforce/codet5-small')

    def __len__(self):
        return len(self.functions_ds)
    
    def __getitem__(self, idx):
        ds_item = self.functions_ds[idx]
        
        input_str = 'complete: ' + ds_item['declaration']
        output_str = ds_item['body']

        input_tokens = self.tokenizer(input_str, return_tensors="pt", padding="max_length", max_length=512, truncation=True)
        output_tokens = self.tokenizer(output_str, return_tensors="pt", padding="max_length", max_length=512, truncation=True)

        return {
            'input_ids': input_tokens['input_ids'].squeeze(),
            'attention_mask': input_tokens['attention_mask'].squeeze(),
            'labels': output_tokens['input_ids'].squeeze(),
            'labels_attention_mask': output_tokens['attention_mask'].squeeze()
        }

def train():
    model = T5ForConditionalGeneration.from_pretrained('./results/final').to(device)
    ds = FunctionDataset()

    training_args = TrainingArguments(
        output_dir=f'{results_path}/stage_2',
        num_train_epochs=3,
        per_device_train_batch_size=16,
        save_strategy='steps',
        save_steps=50000,
        weight_decay=0.01,
        logging_dir=f'{logs_path}/stage_2',
        logging_steps=100,
        evaluation_strategy='no',
        learning_rate=11e-6,
        lr_scheduler_type="cosine",
        metric_for_best_model="loss",
        greater_is_better=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds,
    )

    trainer.train()
    trainer.save_model(f'{results_path}/stage_2/final')

train()
