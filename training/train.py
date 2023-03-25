import os
import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration, TrainingArguments, DataCollatorForLanguageModeling, Trainer
from datasets import Dataset
from itertools import chain
import logging
import db

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"

ds_path = '/home/stepan/ml/solidity-copilot/dataset'
cache_path = './cache'
results_path = './results'
logs_path = './logs'
SEGMENT_SIZE = 10000

device = torch.device('cuda')

logging.getLogger(
    "transformers.tokenization_utils_base").setLevel(logging.ERROR)

BLOCK_SIZE = 512

def prepare_segment(tokenizer: AutoTokenizer, segment_size: int, segment_i: int):
    def tokenize_function(tokenizer):
        def f(item):
            return tokenizer(item['source_code'], return_tensors="pt", padding=True)

        return f
    
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {
            k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= BLOCK_SIZE:
            total_length = (total_length // BLOCK_SIZE) * BLOCK_SIZE
        # Split by chunks of max_len.
        result = {
            k: [t[i: i + BLOCK_SIZE] for i in range(0, total_length, BLOCK_SIZE)]
            for k, t in concatenated_examples.items()
        }
        return result

    segment_contracts = list(map(lambda x: { 'source_code': x['source_code'] }, db.contracts_collection.find({}, limit=segment_size, skip=(segment_i * segment_size))))
    segment_ds = Dataset.from_list(
        segment_contracts)

    tokenized = segment_ds.map(
        tokenize_function(tokenizer),
        remove_columns=['source_code'],
        batched=True,
        batch_size=5,
        num_proc=6,
        desc=f'Segment #{segment_i} | Tokenizing'
    )
    
    return tokenized.map(
        group_texts,
        batched=True,
        batch_size=5,
        num_proc=6,
        drop_last_batch=True,
        desc=f'Segment #{segment_i} | Grouping into batches'
    )


def train(model, tokenizer, segment_size: int, segment_i: int, meta_i: int):
    segment_ds_tokenized = prepare_segment(
        tokenizer, segment_size, segment_i).train_test_split(0.01)

    training_args = TrainingArguments(
        output_dir=f'{results_path}/meta-{meta_i}/segment-{segment_i}',
        num_train_epochs=1,
        per_device_train_batch_size=6,
        eval_accumulation_steps=5,
        save_strategy='epoch',
        weight_decay=0.01,
        logging_dir=f'{logs_path}/meta-{meta_i}/segment-{segment_i}',
        logging_steps=100,
        evaluation_strategy='epoch',
        learning_rate=11e-6,
        lr_scheduler_type="cosine"
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=segment_ds_tokenized["train"],
        eval_dataset=segment_ds_tokenized["test"],
        data_collator=data_collator,
        
    )

    # trainer.evaluate()
    trainer.train()

# checkpoint = 'Salesforce/codet5-small'
checkpoint = f'{results_path}/meta-0/segment-46/checkpoint-11131'
tokenizer = AutoTokenizer.from_pretrained('Salesforce/codet5-small')
model = T5ForConditionalGeneration.from_pretrained(checkpoint).to(device)

prepare_segment(tokenizer, 750000, 0)
# for meta_i in range(0, 1):
#     for segment_i in range(47, 75):
#         train(model, tokenizer, SEGMENT_SIZE, segment_i, meta_i)
#         torch.cuda.empty_cache()