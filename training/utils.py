from transformers import T5Tokenizer, T5ForConditionalGeneration, TrainingArguments, DataCollatorForLanguageModeling, Trainer
from datasets import Dataset
import evaluate
from itertools import chain
import logging
import numpy as np

logging.getLogger(
    "transformers.tokenization_utils_base").setLevel(logging.ERROR)


def tokenize_function(tokenizer):
    def f(item):
        return tokenizer(item['text'], return_tensors="pt")

    return f


BLOCK_SIZE = 512


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


def prepare_segment(tokenizer: T5Tokenizer, full_ds: Dataset, segment_size: int, segment_i: int):
    segment_ds = Dataset.from_dict(
        full_ds[segment_i * segment_size:(segment_i + 1) * segment_size])

    return segment_ds.map(
        tokenize_function(tokenizer),
        remove_columns='text',
        batched=True,
        batch_size=5,
        num_proc=6,
        keep_in_memory=True,
        desc=f'Segment #{id} | Tokenizing'
    ).map(
        group_texts,
        batched=True,
        batch_size=5,
        num_proc=6,
        keep_in_memory=True,
        desc=f'Segment #{id} | Grouping into batches'
    )


def compute_metrics(eval_preds):
    metric = evaluate.load("glue", "mrpc")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def train(checkpoint: str, ds: Dataset, segment_size: int, segment_i: int, meta_i: int, device):
    tokenizer = T5Tokenizer.from_pretrained(checkpoint)
    model = T5ForConditionalGeneration.from_pretrained(checkpoint).to(device)

    segment_ds_tokenized = prepare_segment(
        tokenizer, ds, segment_size, segment_i).train_test_split(0.05)

    training_args = TrainingArguments(
        output_dir=f'./results/meta-epoch-{meta_i}/segment-{segment_i}',
        num_train_epochs=5,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        warmup_steps=500,
        save_steps=50000,
        weight_decay=0.01,
        logging_dir=f'./logs/meta-epoch-{meta_i}/segment-{segment_i}',
        logging_steps=10,
        evaluation_strategy='epoch',
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=segment_ds_tokenized["train"],
        eval_dataset=segment_ds_tokenized["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    trainer.train()
