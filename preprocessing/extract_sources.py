from datasets import Dataset
from db import contracts_collection, libraries_collection, interfaces_collection
from parsing import process_source



def save_data(data):
    for contract in data['contracts']:
        if contracts_collection.find_one({"md5_hash": contract["md5_hash"]}) is None:
            contracts_collection.insert_one(contract)
        # else:
        #     print("duplicate contract", contract["name"])
    for library in data['libraries']:
        if libraries_collection.find_one({"md5_hash": library["md5_hash"]}) is None:
            libraries_collection.insert_one(library)
        # else:
        #     print("duplicate library", library["name"])
    for interface in data['interfaces']:
        if interfaces_collection.find_one({"md5_hash": interface["md5_hash"]}) is None:
            interfaces_collection.insert_one(interface)
        # else:
        #     print("duplicate interface", interface["name"])

def mapper(item):
    sample_parsed = process_source(item["text"])
    save_data(sample_parsed)
    return {}

ds_path = '../dataset'
Dataset.load_from_disk(ds_path).map(
    mapper,
    num_proc=6,
    desc=f'Preprocessing'
)

# text = """
# contract Name is ERC20("one", 2) {
#     address o;
# }
# """
# print(process_source(text))