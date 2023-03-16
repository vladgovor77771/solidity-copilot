import os
import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration, TrainingArguments, DataCollatorForLanguageModeling, Trainer
from datasets import Dataset
import evaluate
from itertools import chain
import logging

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"

ds_path = '/home/stepan/ml/solidity-copilot/dataset'
cache_path = './cache'
results_path = './results'
logs_path = './logs'
SEGMENT_SIZE = 1000

device = torch.device('cuda')
ds = Dataset.load_from_disk(ds_path).filter(lambda x: len(x['text']) > 60)

logging.getLogger(
    "transformers.tokenization_utils_base").setLevel(logging.ERROR)

def tokenize_function(tokenizer):
    def f(item):
        return tokenizer(item['text'], return_tensors="pt", padding=True)

    return f

BLOCK_SIZE = 512


def prepare_segment(tokenizer: AutoTokenizer, full_ds: Dataset, segment_size: int, segment_i: int):
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

    segment_ds = Dataset.from_dict(
        full_ds[segment_i * segment_size:(segment_i + 1) * segment_size])

    tokenized = segment_ds.map(
        tokenize_function(tokenizer),
        remove_columns='text',
        batched=True,
        batch_size=5,
        num_proc=6,
        cache_file_name=f'{cache_path}/seg_{segment_i}_tokenize_map_cache.cache',
        # load_from_cache_file=True,
        desc=f'Segment #{segment_i} | Tokenizing'
    )
    
    return tokenized.map(
        group_texts,
        batched=True,
        batch_size=5,
        num_proc=6,
        drop_last_batch=True,
        cache_file_name=f'{cache_path}/seg_{segment_i}_group_map_cache.cache',
        # load_from_cache_file=True,
        desc=f'Segment #{segment_i} | Grouping into batches'
    )


def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        # Depending on the model and config, logits may contain extra tensors,
        # like past_key_values, but logits always come first
        logits = logits[0]
    return logits.argmax(dim=-1)


metric = evaluate.load("accuracy")

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    labels = labels[:, 1:].reshape(-1)
    preds = preds[:, :-1].reshape(-1)
    return metric.compute(predictions=preds, references=labels)


def train(model, tokenizer, ds: Dataset, segment_size: int, segment_i: int, meta_i: int):
    segment_ds_tokenized = prepare_segment(
        tokenizer, ds, segment_size, segment_i).train_test_split(0.01)

    training_args = TrainingArguments(
        output_dir=f'{results_path}/meta-{meta_i}/segment-{segment_i}',
        num_train_epochs=1,
        per_device_train_batch_size=6,
        per_device_eval_batch_size=4,
        eval_accumulation_steps=5,
        save_strategy='epoch',
        weight_decay=0.01,
        logging_dir=f'{logs_path}/meta-{meta_i}/segment-{segment_i}',
        logging_steps=100,
        evaluation_strategy='epoch',
        # fp16=True,
        learning_rate=12e-7
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False, mlm_probability=0.15)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=segment_ds_tokenized["train"],
        eval_dataset=segment_ds_tokenized["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )

    # trainer.evaluate()
    trainer.train()

checkpoint = 'Salesforce/codet5-small'
# checkpoint = './results/after-segment-4'
tokenizer = AutoTokenizer.from_pretrained('Salesforce/codet5-small')
model = T5ForConditionalGeneration.from_pretrained(checkpoint).to(device)

for meta_i in range(0, 3):
    for segment_i in range(0, len(ds) // SEGMENT_SIZE + (1 if len(ds) % SEGMENT_SIZE != 0 else 0)):
        train(model, tokenizer, ds, SEGMENT_SIZE, segment_i, meta_i)
        torch.cuda.empty_cache()