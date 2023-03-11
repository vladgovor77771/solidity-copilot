import re
import glob
import itertools
import asyncio
import aiofiles
import sys

assert len(sys.args) == 2, "Args: path_to_raw_ds_folder"
raw_ds_folder = sys.args[1]


def remove_comments(string):
    pattern = r"(\".*?\"|\'.*?\')|(/\*.*?\*/|//[^\r\n]*$)"
    # first group captures quoted strings (double or single)
    # second group captures comments (//single-line or /* multi-line */)
    regex = re.compile(pattern, re.MULTILINE | re.DOTALL)

    def _replacer(match):
        # if the 2nd group (capturing comments) is not None,
        # it means we have captured a non-quoted (real) comment string.
        if match.group(2) is not None:
            return ""  # so we will return empty to remove the comment
        else:  # otherwise, we will return the 1st group
            return match.group(1)  # captured quoted-string
    return regex.sub(_replacer, string)


def replace_addr(src_in):
    return re.sub("0x[A-Fa-f0-9]{40}", "YOUR_ADDR", src_in)


def format(src):
    src = remove_comments(src)
    # src = replace_addr(src)
    return src


paths = glob.glob(f'./{raw_ds_folder}/*.sol')
cnt = itertools.count()


async def work():
    while True:
        cur = next(cnt)
        if cur >= len(paths):
            break
        try:
            async with aiofiles.open(paths[cur], "r") as f:
                txt = await f.read()

            formatted = format(txt)
            if formatted != txt:
                async with aiofiles.open(paths[cur], "w") as f:
                    await f.write(formatted)

            # cmd = ["npx", "prettier", "--write", paths[cur]]
            # process = await asyncio.create_subprocess_shell(
            #     " ".join(cmd),
            #     stdout=asyncio.subprocess.PIPE,
            #     stderr=asyncio.subprocess.PIPE)
            # await process.communicate()

        except Exception as e:
            print('Error for file', paths[cur], e)

        if cur != 0 and cur % 1000 == 0:
            print('done for', cur)

loop = asyncio.new_event_loop()
loop.run_until_complete(asyncio.wait([work() for _ in range(32)]))
loop.close()
