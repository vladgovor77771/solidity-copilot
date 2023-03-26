from db import functions_collections
from datasets import Dataset

functions = functions_collections.find({}, limit=900000)
ds = Dataset.from_list(list(map(lambda x: {'declaration': x['declaration'], 'body': x['body']}, functions)))

ds.save_to_disk('../functions_ds')
