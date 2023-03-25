from transformers import AutoTokenizer
# from datasets import Dataset
# from itertools import chain
# import logging

# logging.getLogger(
#     "transformers.tokenization_utils_base").setLevel(logging.ERROR)
tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-large")

print(tokenizer.all_special_tokens_extended)
# def tokenize_function(item):
#     return tokenizer(item['text'], return_tensors="pt", padding=True)


# block_size = 512


# def group_texts(examples):
#     # Concatenate all texts.
#     concatenated_examples = {
#         k: list(chain(*examples[k])) for k in examples.keys()}
#     total_length = len(concatenated_examples[list(examples.keys())[0]])
#     # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
#     # customize this part to your needs.
#     if total_length >= block_size:
#         total_length = (total_length // block_size) * block_size
#     # Split by chunks of max_len.
#     result = {
#         k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
#         for k, t in concatenated_examples.items()
#     }
#     return result


# def process_segment(ds: Dataset, id):
#     result = ds.map(
#         tokenize_function,
#         remove_columns='text',
#         batched=True,
#         batch_size=5,
#         num_proc=6,
#         keep_in_memory=True,
#         desc=f'Segment #{id} | Tokenizing'
#     ).map(
#         group_texts,
#         batched=True,
#         batch_size=5,
#         num_proc=6,
#         keep_in_memory=True,
#         desc=f'Segment #{id} | Grouping into batches'
#     )
#     print(f'Size of segment #{id} = {len(result)}')
#     result.save_to_disk(f'./tokenized_segments/seg-{id}')


# SEGMENT_SIZE = 25000  # count of files for batched segment
# ds = Dataset.load_from_disk('./dataset')

# total_segments = len(ds) // SEGMENT_SIZE + (1 if len(ds) %
#                                             SEGMENT_SIZE != 0 else 0)

# for i in range(13, total_segments):
#     print('Processing segment', i)
#     segment = Dataset.from_dict(ds[i * SEGMENT_SIZE:(i + 1) * SEGMENT_SIZE])
#     process_segment(segment, i)
