import os 
import glob
from random import sample, shuffle
import cv2
from PIL import Image
import numpy as np
from pip import main
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt

from copypaste import CopyPaste
from copypaste import copypaste_batch

def imshow(img):
     npimg = img.numpy()
     plt.imshow(np.transpose(npimg, (1, 2, 0)))
     plt.show()


def save_to_png(tensor, path, name):
     arr = np.transpose(tensor.numpy(), (1, 2, 0))

     if not os.path.isdir(path):
          os.makedirs(path)
     plt.imsave(os.path.join(path, name), arr)
     print("saved")


def save_output_dict(output_dict):
     batch_size = len(output_dict["images"])
     path = os.path.join(".", "forks", "composer", "composer", "algorithms", "copypaste", "files", "out", "no_jittering", "output_dict")
     
     for i in range(batch_size):
          save_to_png(output_dict["images"][i], os.path.join(path, str(i), "image"), str(i)+".png")

          for j, mask in enumerate(output_dict["masks"][i]):
               save_to_png(mask, os.path.join(path, str(i), "masks"), str(i)+ "_" + str(j) + ".png")




main_path = os.path.join(".", "forks", "composer", "composer", "algorithms", "copypaste", "files", "examples", "crevasse", "data")
masks_path = os.path.join(main_path, "masks")
image_path = os.path.join(main_path, "images")

img_h = 480
img_l = 520
# batch_size = 15

input_dict = {
     "sample_names": [],
     "masks": [],
     "images": []
}

sample_names = [name for name in os.listdir(main_path) if os.path.isdir(os.path.join(main_path, name))]

trns = transforms.Compose([transforms.Resize((img_h, img_l)), transforms.ToTensor()])


for sample_name in sample_names:
     sample_path = os.path.join(main_path, sample_name)
     masks_path = os.path.join(sample_path, "masks")
     num_instances = len([name for name in os.listdir(masks_path) if name[-3:] == "png"]) + 1
     # print([name for name in os.listdir(masks_path)])

     data = datasets.ImageFolder(sample_path, transform=trns)
     data_loader = torch.utils.data.DataLoader(data, batch_size=num_instances, shuffle=False)

     for i, data in enumerate(data_loader):
          # print(sample_name, ", i=", i , " -- ", len(data), " -- ", data[0].shape)
          # print(x[1])
          # imshow(torchvision.utils.make_grid(x[0], nrow=5))
          # print("-------------")
          # imshow(data[0][0])

          input_dict["sample_names"].append(sample_name)
          input_dict["masks"].append(data[0][1:])
          input_dict["images"].append(data[0][0])

# data_iter = iter(data_loader)
# images_tensor, _ = data_iter.next()
# masks_tensor, _ = data_iter.next()

# for image in input["images"]:
#      imshow(image)

output_dict = copypaste_batch(input_dict)

save_output_dict(output_dict)




print("done")







