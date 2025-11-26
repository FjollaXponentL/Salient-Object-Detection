import os, random
random.seed(42)

DATA_ROOT = "data/ECSSD"
IMGS = os.path.join(DATA_ROOT, "images")
OUT = DATA_ROOT  # do krijojë train.txt, val.txt, test.txt këtu

files = sorted([f for f in os.listdir(IMGS) if f.lower().endswith(('.jpg','.jpeg','.png'))])
n = len(files)
n_train = int(0.7 * n)
n_val = int(0.15 * n)

random.shuffle(files)
train = files[:n_train]
val = files[n_train:n_train+n_val]
test = files[n_train+n_val:]

open(os.path.join(OUT, "train.txt"), "w").write("\n".join(train))
open(os.path.join(OUT, "val.txt"), "w").write("\n".join(val))
open(os.path.join(OUT, "test.txt"), "w").write("\n".join(test))

print("Lists created in", OUT)
print("Train:", len(train), "Val:", len(val), "Test:", len(test))
