import time
from io import BytesIO

from PIL import Image
from torch.utils.data import IterableDataset

from composer.datasets.streaming import StreamingDataset


class ADE20KDataset(StreamingDataset):

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        uid = sample["uid"].decode("utf-8")
        image = Image.open(BytesIO(sample["image"]))
        annotation = Image.open(BytesIO(sample["annotation"]))
        return {"uid": uid, "image": image, "annotation": annotation}


split = "train"
# remote = f"s3://mds-ade20k/{split}"
remote = f"./mds-ADE20K/{split}"
local = f"/tmp/mds-ADE20K/{split}"
shuffle = False

ds = ADE20KDataset(remote=remote, local=local, shuffle=shuffle)

for epoch in range(3):
    c = 0
    start = time.time()
    for ix, sample in enumerate(ds):
        if (ix < 3):
            print(sample)
        c += 1
    end = time.time()
    sps = c / (end - start)
    print(f"epoch={epoch}, samples={c}, sps={sps:.2f}")
