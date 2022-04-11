import time
from io import BytesIO

from PIL import Image
from torch.utils.data import IterableDataset, DataLoader
from torchvision import transforms
from composer.datasets.streaming import StreamingDataset


#class ADE20KDataset(StreamingDataset):
#
#    def __getitem__(self, idx):
#        sample = super().__getitem__(idx)
#        uid = sample["uid"].decode("utf-8")
#        image = Image.open(BytesIO(sample["image"]))
#        annotation = Image.open(BytesIO(sample["annotation"]))
#        return {"uid": uid, "image": image, "annotation": annotation}


split = "train"
remote = f"s3://mds-ade20k/{split}"
#remote = f"./mds-ADE20K/{split}"
local = f"/tmp/mds-ADE20K/{split}"
shuffle = False
transform = transforms.Compose([
  transforms.RandomCrop(512, 512),
  transforms.ToTensor(),
])
decoders = {
  "uid": lambda uid: uid.decode("utf-8"),
  "image": lambda image: transform(Image.open(BytesIO(image))),
  "annotation": lambda annotation: transform(Image.open(BytesIO(annotation))),
}

BS = 4

dataset = StreamingDataset(remote=remote, local=local, decoders=decoders, shuffle=shuffle, device_batch_size=BS)

loader = DataLoader(
  dataset,
  batch_size=BS,
  num_workers=10,
  prefetch_factor=2,
  persistent_workers=True,
  pin_memory=True,
  drop_last=False,
  timeout=10,
)

print (f"dataset_len={len(dataset)}, dataloader_len={len(loader)}")

for epoch in range(2):
    c = 0
    batch_sizes = {}
    start = time.time()
    for ix, batch in enumerate(loader):
      #if (ix == 0):
      #  print(batch)

      bs = len(batch["uid"])
      c += bs
      
      if bs in batch_sizes:
        batch_sizes[bs] += 1
      else:
        batch_sizes[bs] = 1

      print (ix, c)
    end = time.time()
    sps = c / (end - start)
    print(f"epoch={epoch}, samples={c}, sps={sps:.2f}")
    print(batch_sizes)

