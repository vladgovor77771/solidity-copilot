import glob
import shutil
import os
import sys
from git import Repo

assert len(
    sys.args) == 3, "Args: path to repos.txt and path to folder to collect ds to"
repos_path = sys.args[1]
ds_path = sys.args[2]

cur = 0
repos_done = 0
with open(repos_path, "r") as f:
    for i in range(repos_done):
        f.readline()
    while True:
        repo_url = f.readline().strip()
        if not repo_url:
            break

        try:
            try:
                shutil.rmtree("./tmp")
            except:
                pass

            print("Cloning " + repo_url + "...")
            Repo.clone_from(repo_url, "./tmp")
            print("Cloned, saving .sol files")
            for path in glob.iglob('./tmp/**/*.sol', recursive=True):
                if os.path.isfile(path):
                    shutil.copyfile(path, f"{ds_path}/" + str(cur) + ".sol")
                    cur += 1
            shutil.rmtree("./tmp")
            repos_done += 1
            print("Progress: " + str(repos_done) +
                  ". Total files: " + str(cur))
        except KeyboardInterrupt:
            print("Stop")
            break
        except Exception as e:
            print("Error for " + repo_url)
            print(e)
