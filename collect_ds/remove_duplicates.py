import glob
import hashlib
import os
import sys


assert len(sys.args) == 2, "Args: path_to_raw_ds_folder"
raw_ds_folder = sys.args[1]


def md5(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


hashes = {}
cnt = 0
good = 0

for path in glob.iglob(f'./{raw_ds_folder}/*.sol'):
    hash_ = md5(path)
    if hashes.get(hash_) is None:
        good += 1
        hashes[hash_] = path
    else:
        print("Duplicate " + path + " of " + hashes[hash_])
        cnt += 1
        os.remove(path)

print("Total deleted files " + str(cnt) + ", Good " + str(good))
