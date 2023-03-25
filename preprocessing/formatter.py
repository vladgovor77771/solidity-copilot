from datasets import Dataset
import subprocess
from parsing import extract_functions
from db import functions_collections


def format_with_prettier(source, file_path="Main.sol"):
    process = subprocess.Popen(
        ["npx", "prettier", "--stdin-filepath", file_path],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    stdout, stderr = process.communicate(source)
    
    if process.returncode != 0:
        print(f"Prettier error: {stderr}")
        return source
    
    return stdout.strip()

def save_data(functions):
    for func in functions:
        if functions_collections.find_one({"hash": func["hash"]}) is None:
            functions_collections.insert_one(func)

def mapper(item):
    try:
        formatted = format_with_prettier(item['text'])
        funcs = extract_functions(formatted)
        save_data(funcs)
    except Exception as e:
        pass
    return {}

ds_path = '../dataset'
ds = Dataset.load_from_disk(ds_path)
ds.map(
    mapper,
    num_proc=16,
    desc=f'Formatting'
)
