import asyncio
import glob
import re
import itertools
import sys

assert len(sys.args) == 2, "Args: path_to_raw_ds_folder"
raw_ds_folder = sys.args[1]

deleted_count = itertools.count()


async def remove_invalid_files(stderr):
    pattern = r'\[error\]\s+([^\s]+\s)*[^\s]+\.sol'
    matches = re.findall(pattern, stderr)
    sol_paths = [re.search(r'[^\s]+\.sol', match).group(0)
                 for match in matches]
    for path in sol_paths:
        cnt = next(deleted_count)
        process = await asyncio.create_subprocess_exec("rm", path)
        await process.wait()
        print(f"Deleted already {cnt + 1} files")

chunk = []
chunk_length = 2000
chunks = []

for file in glob.iglob(f'./{raw_ds_folder}/*.sol'):
    chunk.append(file)
    if len(chunk) == chunk_length:
        chunks.append(chunk.copy())
        chunk.clear()
if len(chunk) != 0:
    chunks.append(chunk.copy())
    chunk.clear()

current_chunk = itertools.count()


async def run_worker(index):
    print(f"Running worker #{index}")
    while True:
        cur = next(current_chunk)
        if cur >= len(chunks):
            print(f"Worker {index} stopped")
            break
        chunk = chunks[cur]
        cmd = ["npx", "prettier", "--write"] + chunk
        process = await asyncio.create_subprocess_shell(
            " ".join(cmd),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE)
        _, stderr = await process.communicate()
        if process.returncode != 0:
            await remove_invalid_files(str(stderr))
        print(f"Worker #{index} processed chunk #{cur + 1} of {len(chunks)}")

loop = asyncio.new_event_loop()
loop.run_until_complete(asyncio.wait([run_worker(x) for x in range(8)]))
loop.close()
