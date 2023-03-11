import glob
from progress.bar import IncrementalBar
from datasets import Dataset
import sys

assert len(sys.args) == 3, "Args: path_to_raw_ds_folder, path_to_created_ds"
raw_ds_folder = sys.args[1]
ds_folder = sys.args[2]

paths = glob.glob(f'./{raw_ds_folder}/*.sol')
texts = []
bar = IncrementalBar('Total files', max=len(paths))

for i, path in enumerate(paths):
    bar.next()
    with open(path, 'r') as f:
        texts.append({'text': f.read()})

bar.finish()
ds = Dataset.from_list(texts)
ds.save_to_disk(ds_folder)
