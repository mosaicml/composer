## Code adapted from https://github.com/mlcommons/training/tree/master/single_stage_detector/ssd

import copy
import itertools
import json
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from urllib.request import urlretrieve

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data as data
import yahp as hp
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
from PIL import Image
from pycocotools import mask as maskUtils

from composer.core.types import DataLoader, Dataset, DataSpec
from composer.datasets.dataloader import DataloaderHparams
from composer.datasets.hparams import DatasetHparams
from composer.models.ssd.utils import DefaultBoxes, SSDTransformer


def dboxes300_coco():
    figsize = 300
    feat_size = [38, 19, 10, 5, 3, 1]
    steps = [8, 16, 32, 64, 100, 300]
    # use the scales here: https://github.com/amdegroot/ssd.pytorch/blob/master/data/config.py
    scales = [21, 45, 99, 153, 207, 261, 315]
    aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
    dboxes = DefaultBoxes(figsize, feat_size, steps, scales, aspect_ratios)
    return dboxes


@dataclass
class COCODatasetHparams(DatasetHparams):
    """Defines an instance of 

    Parameters:
        is_train (bool): Whether to load the training or validation dataset.
        datadir (str): Data directory to use.
        download (bool): Whether to download the dataset, if needed.
        drop_last (bool): Whether to drop the last samples for the last batch.
        shuffle (bool): Whether to shuffle the dataset for each epoch.

    """

    is_train: bool = hp.required("whether to load the training or validation dataset")
    drop_last: bool = hp.required("Whether to drop the last samples for the last batch")
    datadir: str = hp.required("data directory")
    shuffle: bool = hp.required("Whether to shuffle the dataset for each epoch")
    download: bool = hp.required("whether to download the dataset, if needed")

    from composer.core.types import DataLoader

    def initialize_object(self, batch_size: int, dataloader_hparams: DataloaderHparams):

        dboxes = dboxes300_coco()

        input_size = 300
        train_trans = SSDTransformer(dboxes, (input_size, input_size), val=False, num_cropping_iterations=1)
        val_trans = SSDTransformer(dboxes, (input_size, input_size), val=True)
        data = "/mnt/cota/datasets/coco"

        val_annotate = os.path.join(data, "annotations/instances_val2017.json")
        val_coco_root = os.path.join(data, "val2017")
        train_annotate = os.path.join(data, "annotations/instances_train2017.json")
        train_coco_root = os.path.join(data, "train2017")

        cocoGt = COCO(annotation_file=val_annotate)
        train_coco = COCODetection(train_coco_root, train_annotate, train_trans)
        val_coco = COCODetection(val_coco_root, val_annotate, val_trans)

        if self.is_train:
            return DataSpec(dataloader=dataloader_hparams.initialize_object(
                dataset=train_coco,
                batch_size=batch_size,
                sampler=None,
                drop_last=self.drop_last,
            ))
        else:
            return DataSpec(dataloader=dataloader_hparams.initialize_object(
                dataset=val_coco,
                drop_last=self.drop_last,
                batch_size=batch_size,
                sampler=None,
            ))


def _isArrayLike(obj):
    return hasattr(obj, '__iter__') and hasattr(obj, '__len__')


class COCODetection(data.Dataset):

    def __init__(self, img_folder, annotate_file, transform=None):
        self.img_folder = img_folder
        self.annotate_file = annotate_file

        # Start processing annotation
        with open(annotate_file) as fin:
            self.data = json.load(fin)

        self.images = {}

        self.label_map = {}
        self.label_info = {}
        start_time = time.time()
        # 0 stands for the background
        cnt = 0
        self.label_info[cnt] = "background"
        for cat in self.data["categories"]:
            cnt += 1
            self.label_map[cat["id"]] = cnt
            self.label_info[cnt] = cat["name"]

        # build inference for images
        for img in self.data["images"]:
            img_id = img["id"]
            img_name = img["file_name"]
            img_size = (img["height"], img["width"])
            if img_id in self.images:
                raise Exception("dulpicated image record")
            self.images[img_id] = (img_name, img_size, [])

        # read bboxes
        for bboxes in self.data["annotations"]:
            img_id = bboxes["image_id"]
            category_id = bboxes["category_id"]
            bbox = bboxes["bbox"]
            bbox_label = self.label_map[bboxes["category_id"]]
            self.images[img_id][2].append((bbox, bbox_label))

        for k, v in list(self.images.items()):
            if len(v[2]) == 0:
                # print("empty image: {}".format(k))
                self.images.pop(k)

        self.img_keys = list(self.images.keys())
        self.transform = transform

    #@property
    def labelnum(self):
        return len(self.label_info)

    @staticmethod
    def load(pklfile):
        with bz2.open(pklfile, "rb") as fin:
            ret = pickle.load(fin)
        return ret

    def save(self, pklfile):
        with bz2.open(pklfile, "wb") as fout:
            pickle.dump(self, fout)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_id = self.img_keys[idx]
        img_data = self.images[img_id]
        fn = img_data[0]
        img_path = os.path.join(self.img_folder, fn)

        from os.path import exists

        if (exists(img_path)):
            img = Image.open(img_path).convert("RGB")

            htot, wtot = img_data[1]
            bbox_sizes = []
            bbox_labels = []

            for (l, t, w, h), bbox_label in img_data[2]:
                r = l + w
                b = t + h
                bbox_size = (l / wtot, t / htot, r / wtot, b / htot)
                bbox_sizes.append(bbox_size)
                bbox_labels.append(bbox_label)

            bbox_sizes = torch.tensor(bbox_sizes)
            bbox_labels = torch.tensor(bbox_labels)

            if self.transform != None:
                img, (htot, wtot), bbox_sizes, bbox_labels = \
                    self.transform(img, (htot, wtot), bbox_sizes, bbox_labels)
            else:
                pass
            return img, img_id, (htot, wtot), bbox_sizes, bbox_labels
        else:
            print('here')
            pass


class COCO:

    def __init__(self, annotation_file=None):
        """
        Constructor of Microsoft COCO helper class for reading and visualizing annotations.
        :param annotation_file (str): location of annotation file
        :param image_folder (str): location to the folder that hosts images.
        :return:
        """
        # load dataset
        self.dataset, self.anns, self.cats, self.imgs = dict(), dict(), dict(), dict()
        self.imgToAnns, self.catToImgs = defaultdict(list), defaultdict(list)
        if not annotation_file == None:
            print('loading annotations into memory...')
            tic = time.time()
            dataset = json.load(open(annotation_file, 'r'))
            assert type(dataset) == dict, 'annotation file format {} not supported'.format(type(dataset))
            print('Done (t={:0.2f}s)'.format(time.time() - tic))
            self.dataset = dataset
            self.createIndex()

    def createIndex(self):
        # create index
        print('creating index...')
        anns, cats, imgs = {}, {}, {}
        imgToAnns, catToImgs = defaultdict(list), defaultdict(list)
        if 'annotations' in self.dataset:
            for ann in self.dataset['annotations']:
                imgToAnns[ann['image_id']].append(ann)
                anns[ann['id']] = ann

        if 'images' in self.dataset:
            for img in self.dataset['images']:
                imgs[img['id']] = img

        if 'categories' in self.dataset:
            for cat in self.dataset['categories']:
                cats[cat['id']] = cat

        if 'annotations' in self.dataset and 'categories' in self.dataset:
            for ann in self.dataset['annotations']:
                catToImgs[ann['category_id']].append(ann['image_id'])

        print('index created!')

        # create class members
        self.anns = anns
        self.imgToAnns = imgToAnns
        self.catToImgs = catToImgs
        self.imgs = imgs
        self.cats = cats

    def info(self):
        """
        Print information about the annotation file.
        :return:
        """
        for key, value in self.dataset['info'].items():
            print('{}: {}'.format(key, value))

    def getAnnIds(self, imgIds=[], catIds=[], areaRng=[], iscrowd=None):
        """
        Get ann ids that satisfy given filter conditions. default skips that filter
        :param imgIds  (int array)     : get anns for given imgs
               catIds  (int array)     : get anns for given cats
               areaRng (float array)   : get anns for given area range (e.g. [0 inf])
               iscrowd (boolean)       : get anns for given crowd label (False or True)
        :return: ids (int array)       : integer array of ann ids
        """
        imgIds = imgIds if _isArrayLike(imgIds) else [imgIds]
        catIds = catIds if _isArrayLike(catIds) else [catIds]

        if len(imgIds) == len(catIds) == len(areaRng) == 0:
            anns = self.dataset['annotations']
        else:
            if not len(imgIds) == 0:
                lists = [self.imgToAnns[imgId] for imgId in imgIds if imgId in self.imgToAnns]
                anns = list(itertools.chain.from_iterable(lists))
            else:
                anns = self.dataset['annotations']
            anns = anns if len(catIds) == 0 else [ann for ann in anns if ann['category_id'] in catIds]
            anns = anns if len(areaRng) == 0 else [
                ann for ann in anns if ann['area'] > areaRng[0] and ann['area'] < areaRng[1]
            ]
        if not iscrowd == None:
            ids = [ann['id'] for ann in anns if ann['iscrowd'] == iscrowd]
        else:
            ids = [ann['id'] for ann in anns]
        return ids

    def getCatIds(self, catNms=[], supNms=[], catIds=[]):
        """
        filtering parameters. default skips that filter.
        :param catNms (str array)  : get cats for given cat names
        :param supNms (str array)  : get cats for given supercategory names
        :param catIds (int array)  : get cats for given cat ids
        :return: ids (int array)   : integer array of cat ids
        """
        catNms = catNms if _isArrayLike(catNms) else [catNms]
        supNms = supNms if _isArrayLike(supNms) else [supNms]
        catIds = catIds if _isArrayLike(catIds) else [catIds]

        if len(catNms) == len(supNms) == len(catIds) == 0:
            cats = self.dataset['categories']
        else:
            cats = self.dataset['categories']
            cats = cats if len(catNms) == 0 else [cat for cat in cats if cat['name'] in catNms]
            cats = cats if len(supNms) == 0 else [cat for cat in cats if cat['supercategory'] in supNms]
            cats = cats if len(catIds) == 0 else [cat for cat in cats if cat['id'] in catIds]
        ids = [cat['id'] for cat in cats]
        return ids

    def getImgIds(self, imgIds=[], catIds=[]):
        '''
        Get img ids that satisfy given filter conditions.
        :param imgIds (int array) : get imgs for given ids
        :param catIds (int array) : get imgs with all given cats
        :return: ids (int array)  : integer array of img ids
        '''
        imgIds = imgIds if _isArrayLike(imgIds) else [imgIds]
        catIds = catIds if _isArrayLike(catIds) else [catIds]

        if len(imgIds) == len(catIds) == 0:
            ids = self.imgs.keys()
        else:
            ids = set(imgIds)
            for i, catId in enumerate(catIds):
                if i == 0 and len(ids) == 0:
                    ids = set(self.catToImgs[catId])
                else:
                    ids &= set(self.catToImgs[catId])
        return list(ids)

    def loadAnns(self, ids=[]):
        """
        Load anns with the specified ids.
        :param ids (int array)       : integer ids specifying anns
        :return: anns (object array) : loaded ann objects
        """
        if _isArrayLike(ids):
            return [self.anns[id] for id in ids]
        elif type(ids) == int:
            return [self.anns[ids]]

    def loadCats(self, ids=[]):
        """
        Load cats with the specified ids.
        :param ids (int array)       : integer ids specifying cats
        :return: cats (object array) : loaded cat objects
        """
        if _isArrayLike(ids):
            return [self.cats[id] for id in ids]
        elif type(ids) == int:
            return [self.cats[ids]]

    def loadImgs(self, ids=[]):
        """
        Load anns with the specified ids.
        :param ids (int array)       : integer ids specifying img
        :return: imgs (object array) : loaded img objects
        """
        if _isArrayLike(ids):
            return [self.imgs[id] for id in ids]
        elif type(ids) == int:
            return [self.imgs[ids]]

    def showAnns(self, anns):
        """
        Display the specified annotations.
        :param anns (array of object): annotations to display
        :return: None
        """
        if len(anns) == 0:
            return 0
        if 'segmentation' in anns[0] or 'keypoints' in anns[0]:
            datasetType = 'instances'
        elif 'caption' in anns[0]:
            datasetType = 'captions'
        else:
            raise Exception('datasetType not supported')
        if datasetType == 'instances':
            ax = plt.gca()
            ax.set_autoscale_on(False)
            polygons = []
            color = []
            for ann in anns:
                c = (np.random.random((1, 3)) * 0.6 + 0.4).tolist()[0]
                if 'segmentation' in ann:
                    if type(ann['segmentation']) == list:
                        # polygon
                        for seg in ann['segmentation']:
                            poly = np.array(seg).reshape((int(len(seg) / 2), 2))
                            polygons.append(Polygon(poly))
                            color.append(c)
                    else:
                        # mask
                        t = self.imgs[ann['image_id']]
                        if type(ann['segmentation']['counts']) == list:
                            rle = maskUtils.frPyObjects([ann['segmentation']], t['height'], t['width'])
                        else:
                            rle = [ann['segmentation']]
                        m = maskUtils.decode(rle)
                        img = np.ones((m.shape[0], m.shape[1], 3))
                        if ann['iscrowd'] == 1:
                            color_mask = np.array([2.0, 166.0, 101.0]) / 255
                        if ann['iscrowd'] == 0:
                            color_mask = np.random.random((1, 3)).tolist()[0]
                        for i in range(3):
                            img[:, :, i] = color_mask[i]
                        ax.imshow(np.dstack((img, m * 0.5)))
                if 'keypoints' in ann and type(ann['keypoints']) == list:
                    # turn skeleton into zero-based index
                    sks = np.array(self.loadCats(ann['category_id'])[0]['skeleton']) - 1
                    kp = np.array(ann['keypoints'])
                    x = kp[0::3]
                    y = kp[1::3]
                    v = kp[2::3]
                    for sk in sks:
                        if np.all(v[sk] > 0):
                            plt.plot(x[sk], y[sk], linewidth=3, color=c)
                    plt.plot(x[v > 0],
                             y[v > 0],
                             'o',
                             markersize=8,
                             markerfacecolor=c,
                             markeredgecolor='k',
                             markeredgewidth=2)
                    plt.plot(x[v > 1],
                             y[v > 1],
                             'o',
                             markersize=8,
                             markerfacecolor=c,
                             markeredgecolor=c,
                             markeredgewidth=2)
            p = PatchCollection(polygons, facecolor=color, linewidths=0, alpha=0.4)
            ax.add_collection(p)
            p = PatchCollection(polygons, facecolor='none', edgecolors=color, linewidths=2)
            ax.add_collection(p)
        elif datasetType == 'captions':
            for ann in anns:
                print(ann['caption'])

    def loadRes(self, resFile):
        """
        Load result file and return a result api object.
        :param   resFile (str)     : file name of result file
        :return: res (obj)         : result api object
        """
        res = COCO()
        res.dataset['images'] = [img for img in self.dataset['images']]

        print('Loading and preparing results...')
        tic = time.time()
        if type(resFile) == str:  #or type(resFile) == unicode:
            anns = json.load(open(resFile))
        elif type(resFile) == np.ndarray:
            anns = self.loadNumpyAnnotations(resFile)
        else:
            anns = resFile

        #assert type(anns) == list, 'results in not an array of objects'
        #annsImgIds = [ann['image_id'] for ann in anns]
        #assert set(annsImgIds) == (set(annsImgIds) & set(self.getImgIds())), \
        #       'Results do not correspond to current coco set'
        if 'caption' in anns[0]:
            imgIds = set([img['id'] for img in res.dataset['images']]) & set([ann['image_id'] for ann in anns])
            res.dataset['images'] = [img for img in res.dataset['images'] if img['id'] in imgIds]
            for id, ann in enumerate(anns):
                ann['id'] = id + 1
        elif 'bbox' in anns[0] and not anns[0]['bbox'] == []:
            res.dataset['categories'] = copy.deepcopy(self.dataset['categories'])
            for id, ann in enumerate(anns):
                bb = ann['bbox']
                x1, x2, y1, y2 = [bb[0], bb[0] + bb[2], bb[1], bb[1] + bb[3]]
                if not 'segmentation' in ann:
                    ann['segmentation'] = [[x1, y1, x1, y2, x2, y2, x2, y1]]
                ann['area'] = bb[2] * bb[3]
                ann['id'] = id + 1
                ann['iscrowd'] = 0
        elif 'segmentation' in anns[0]:
            res.dataset['categories'] = copy.deepcopy(self.dataset['categories'])
            for id, ann in enumerate(anns):
                # now only support compressed RLE format as segmentation results
                ann['area'] = maskUtils.area(ann['segmentation'])
                if not 'bbox' in ann:
                    ann['bbox'] = maskUtils.toBbox(ann['segmentation'])
                ann['id'] = id + 1
                ann['iscrowd'] = 0
        elif 'keypoints' in anns[0]:
            res.dataset['categories'] = copy.deepcopy(self.dataset['categories'])
            for id, ann in enumerate(anns):
                s = ann['keypoints']
                x = s[0::3]
                y = s[1::3]
                x0, x1, y0, y1 = np.min(x), np.max(x), np.min(y), np.max(y)
                ann['area'] = (x1 - x0) * (y1 - y0)
                ann['id'] = id + 1
                ann['bbox'] = [x0, y0, x1 - x0, y1 - y0]
        print('DONE (t={:0.2f}s)'.format(time.time() - tic))

        res.dataset['annotations'] = anns
        res.createIndex()
        return res

    def download(self, tarDir=None, imgIds=[]):
        '''
        Download COCO images from mscoco.org server.
        :param tarDir (str): COCO results directory name
               imgIds (list): images to be downloaded
        :return:
        '''
        if tarDir is None:
            print('Please specify target directory')
            return -1
        if len(imgIds) == 0:
            imgs = self.imgs.values()
        else:
            imgs = self.loadImgs(imgIds)
        N = len(imgs)
        if not os.path.exists(tarDir):
            os.makedirs(tarDir)
        for i, img in enumerate(imgs):
            tic = time.time()
            fname = os.path.join(tarDir, img['file_name'])
            if not os.path.exists(fname):
                urlretrieve(img['coco_url'], fname)
            print('downloaded {}/{} images (t={:0.1f}s)'.format(i, N, time.time() - tic))

    def loadNumpyAnnotations(self, data):
        """
        Convert result data from a numpy array [Nx7] where each row contains {imageID,x1,y1,w,h,score,class}
        :param  data (numpy.ndarray)
        :return: annotations (python nested list)
        """
        print('Converting ndarray to lists...')
        print('data shape', data.shape)
        N = data.shape[0]
        ann = []
        for i in range(N):
            m = len(data[i])
            for j in range(m):
                ann += [{
                    'image_id': int(data[i][j][0]),
                    'bbox': [data[i][j][1], data[i][j][2], data[i][j][3], data[i][j][4]],
                    'score': data[i][j][5],
                    'category_id': int(data[i][j][6]),
                }]
        return ann

    def annToRLE(self, ann):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        t = self.imgs[ann['image_id']]
        h, w = t['height'], t['width']
        segm = ann['segmentation']
        if type(segm) == list:
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, h, w)
            rle = maskUtils.merge(rles)
        elif type(segm['counts']) == list:
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, h, w)
        else:
            # rle
            rle = ann['segmentation']
        return rle

    def annToMask(self, ann):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        rle = self.annToRLE(ann)
        m = maskUtils.decode(rle)
        return m
