from datasets import Dataset

ds = Dataset.load_from_disk('../functions_ds')
print(len(ds))