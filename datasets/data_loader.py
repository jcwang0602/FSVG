# -*- coding: utf-8 -*-

"""
ReferIt, UNC, UNC+ and GRef referring image segmentation PyTorch dataset.

Define and group batches of images, segmentations and queries.
Based on:
https://github.com/chenxi116/TF-phrasecut-public/blob/master/build_batches.py
"""

import re
import torch
import numpy as np
import os.path as osp
import torch.utils.data as data
from PIL import Image
from transformers import AutoTokenizer
from utils.word_utils import Corpus

import clip


def read_examples(input_line, unique_id):
    """Read a list of `InputExample`s from an input file."""
    examples = []
    # unique_id = 0
    line = input_line  # reader.readline()
    line = line.strip()
    text_a = None
    text_b = None
    m = re.match(r"^(.*) \|\|\| (.*)$", line)
    if m is None:
        text_a = line
    else:
        text_a = m.group(1)
        text_b = m.group(2)
    examples.append(InputExample(unique_id=unique_id, text_a=text_a, text_b=text_b))
    # unique_id += 1
    return examples


## Bert text encoding
class InputExample(object):
    def __init__(self, unique_id, text_a, text_b):
        self.unique_id = unique_id
        self.text_a = text_a
        self.text_b = text_b


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, unique_id, tokens, input_ids, input_mask, input_type_ids):
        self.unique_id = unique_id
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids


def convert_examples_to_features(examples, seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""
    features = []
    for ex_index, example in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > seq_length - 2:
                tokens_a = tokens_a[0 : (seq_length - 2)]
        tokens = []
        input_type_ids = []
        tokens.append("[CLS]")
        input_type_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            input_type_ids.append(0)
        tokens.append("[SEP]")
        input_type_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                input_type_ids.append(1)
            tokens.append("[SEP]")
            input_type_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < seq_length:
            input_ids.append(0)
            input_mask.append(0)
            input_type_ids.append(0)

        assert len(input_ids) == seq_length
        assert len(input_mask) == seq_length
        assert len(input_type_ids) == seq_length
        features.append(InputFeatures(unique_id=example.unique_id, tokens=tokens, input_ids=input_ids, input_mask=input_mask, input_type_ids=input_type_ids))
    return features


class DatasetNotFoundError(Exception):
    pass


class TransVGDataset(data.Dataset):
    SUPPORTED_DATASETS = {
        "referit": {"splits": ("train", "val", "trainval", "test", "train_pseudo")},
        "unc": {"splits": ("train", "val", "trainval", "testA", "testB", "train_pseudo"), "params": {"dataset": "refcoco", "split_by": "unc"}},
        "unc+": {"splits": ("train", "val", "trainval", "testA", "testB", "train_pseudo"), "params": {"dataset": "refcoco+", "split_by": "unc"}},
        "gref": {"splits": ("train", "val", "train_pseudo"), "params": {"dataset": "refcocog", "split_by": "google"}},
        "gref_umd": {"splits": ("train", "val", "test", "train_pseudo"), "params": {"dataset": "refcocog", "split_by": "umd"}},
        "flickr": {"splits": ("train", "val", "test", "train_pseudo")},
    }

    """ the core part of the dataset processing """

    def __init__(self, data_root, split_root="data", dataset="referit", transform=None, return_idx=False, testmode=False, split="train", max_query_len=128, prompt_template=None, lstm=False, bert_model="bert-base-uncased"):
        self.images = []
        self.data_root = data_root
        self.split_root = split_root
        self.dataset = dataset
        self.query_len = max_query_len
        self.lstm = lstm
        self.prompt_template = prompt_template
        self.transform = transform
        self.testmode = testmode
        self.split = split
        # self.tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=True)
        self.return_idx = return_idx

        assert self.transform is not None

        if split in ["train", "train_pseudo"]:
            self.augment = True
        else:
            self.augment = False

        if self.dataset == "referit":
            self.dataset_root = osp.join(self.data_root, "referit")
            self.im_dir = osp.join(self.dataset_root, "images")
            self.split_dir = osp.join(self.dataset_root, "splits")
        elif self.dataset == "flickr":
            self.dataset_root = osp.join(self.data_root, "Flickr30k")
            # TODO: it should be note that this needs to change flickr30k_images to flickr30k-images
            self.im_dir = osp.join(self.dataset_root, "flickr30k-images")
        else:  ## refcoco, etc.
            self.im_dir = osp.join(self.data_root, "mscoco", "images", "train2014")
            self.split_dir = osp.join(self.data_root, "splits")

        if not self.exists_dataset():
            print("The dataset {} is not found!".format(osp.join(self.split_root, self.dataset)))
            print(
                "Please download index cache to data folder: \n \
                https://drive.google.com/open?id=1cZI562MABLtAzM6YU4WmKPFFguuVr0lZ"
            )
            exit(0)

        dataset_path = osp.join(self.split_root, self.dataset)
        valid_splits = self.SUPPORTED_DATASETS[self.dataset]["splits"]

        if self.lstm:
            self.corpus = Corpus()
            corpus_path = osp.join(dataset_path, "corpus.pth")
            self.corpus = torch.load(corpus_path)

        if split not in valid_splits:
            raise ValueError("Dataset {0} does not have split {1}".format(self.dataset, split))

        splits = [split]
        if self.dataset != "referit":
            splits = ["train", "val"] if split == "trainval" else [split]
        for split in splits:
            imgset_file = "{0}_{1}.pth".format(self.dataset, split)
            imgset_path = osp.join(dataset_path, imgset_file)
            self.images += torch.load(imgset_path)

        if self.prompt_template:
            self.images = self.prompt(self.images)

    def exists_dataset(self):
        return osp.exists(osp.join(self.split_root, self.dataset))

    def pull_item(self, idx):
        if self.dataset == "flickr":
            img_file, bbox, phrase = self.images[idx]
        else:
            img_file, _, bbox, phrase, attri = self.images[idx]
        ## box format: to x1y1x2y2
        bbox_ori = bbox
        if not (self.dataset == "referit" or self.dataset == "flickr"):
            bbox = np.array(bbox, dtype=int)
            bbox[2], bbox[3] = bbox[0] + bbox[2], bbox[1] + bbox[3]
        else:
            bbox = np.array(bbox, dtype=int)

        img_path = osp.join(self.im_dir, img_file)
        img = Image.open(img_path).convert("RGB")

        bbox = torch.tensor(bbox)
        bbox = bbox.float()
        return img_file, img, phrase, bbox, bbox_ori

    def tokenize_phrase(self, phrase):
        return self.corpus.tokenize(phrase, self.query_len)

    def untokenize_word_vector(self, words):
        return self.corpus.dictionary[words]

    def prompt(self, sample_list):
        n = len(sample_list)
        new_sample_list = []

        for i in range(n):
            if self.dataset == "flickr":
                tmp_sample = (sample_list[i][0], sample_list[i][1], self.prompt_template.replace("{pseudo_query}", sample_list[i][2]))
            else:
                # print("\nsample_list:\n", sample_list[i])
                #  ('COCO_train2014_000000000839.jpg', '482127.pth', [303.58, 69.03, 293.29, 425.79],
                #  'guy flopping around on the right', [('r1', ['guy']), ('r2', ['none']), ('r3', ['none']),
                #  ('r4', ['none']), ('r5', ['none']), ('r6', ['none']), ('r7', ['none']),
                #  ('r8', ['right', 'flopping', 'around'])])
                tmp_sample = (sample_list[i][0], sample_list[i][1], sample_list[i][2], self.prompt_template.replace("{pseudo_query}", sample_list[i][3]), sample_list[i][4])
            new_sample_list.append(tmp_sample)
        return new_sample_list

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_file, img, phrase, bbox, bbox_ori = self.pull_item(idx)  # bbox  x1y1x2y2
        phrase = phrase.lower()
        input_dict = {"img": img, "box": bbox, "text": phrase}
        input_dict = self.transform(input_dict)
        w, h = img.width, img.height
        img = input_dict["img"]
        img_mask = input_dict["mask"]
        bbox = input_dict["box"]
        phrase = input_dict["text"]

        # text_token = clip.tokenize(phrase, context_length = self.query_len)  # 1*77
        text_token = clip.tokenize(phrase)  # 1*77
        text = text_token.int()[0].tolist()
        text_mask = (text_token.clone() > 0).int()[0].tolist()

        if self.testmode:  # default is False
            return img, np.array(text, dtype=int), np.array(text_mask, dtype=int), np.array(bbox, dtype=np.float32), np.array(ratio, dtype=np.float32), np.array(dw, dtype=np.float32), np.array(dh, dtype=np.float32), self.images[idx][0]
        else:
            return img, np.array(img_mask), np.array(text, dtype=int), np.array(text_mask, dtype=int), np.array(bbox, dtype=np.float32), img_file, phrase, bbox_ori, w, h


class OSVGDataset(data.Dataset):
    SUPPORTED_DATASETS = {
        "referit": {"splits": ("train", "val", "trainval", "test", "train_pseudo")},
        "unc": {"splits": ("train", "val", "trainval", "testA", "testB", "train_pseudo"), "params": {"dataset": "refcoco", "split_by": "unc"}},
        "unc+": {"splits": ("train", "val", "trainval", "testA", "testB", "train_pseudo"), "params": {"dataset": "refcoco+", "split_by": "unc"}},
        "gref": {"splits": ("train", "val", "train_pseudo"), "params": {"dataset": "refcocog", "split_by": "google"}},
        "gref_umd": {"splits": ("train", "val", "test", "train_pseudo"), "params": {"dataset": "refcocog", "split_by": "umd"}},
        "flickr": {"splits": ("train", "val", "test", "train_pseudo")},
    }

    """ the core part of the dataset processing """

    def __init__(self, data_root, split_root="data", dataset="referit", tokenizer_type="bert", transform=None, return_idx=False, testmode=False, split="train", max_query_len=128, prompt_template=None, lstm=False, bert_model="bert-base-uncased"):
        self.images = []
        self.data_root = data_root
        self.split_root = split_root
        self.dataset = dataset
        self.query_len = max_query_len
        self.lstm = lstm
        self.prompt_template = prompt_template
        self.transform = transform
        self.testmode = testmode
        self.split = split
        if tokenizer_type == "roberta":
            self.tokenizer = AutoTokenizer.from_pretrained("pretrained_models/models-FacebookAI-roberta-base")
        elif tokenizer_type == "bert":
            self.tokenizer = AutoTokenizer.from_pretrained("pretrained_models/models-google-bert-bert-base-uncased")
        self.return_idx = return_idx

        assert self.transform is not None

        if split in ["train", "train_pseudo"]:
            self.augment = True
        else:
            self.augment = False

        if self.dataset == "referit":
            self.dataset_root = osp.join(self.data_root, "referit")
            self.im_dir = osp.join(self.dataset_root, "images")
            self.split_dir = osp.join(self.dataset_root, "splits")
        elif self.dataset == "flickr":
            self.dataset_root = osp.join(self.data_root, "Flickr30k")
            # TODO: it should be note that this needs to change flickr30k_images to flickr30k-images
            self.im_dir = osp.join(self.dataset_root, "flickr30k-images")
        else:  ## refcoco, etc.
            self.im_dir = osp.join(self.data_root, "mscoco", "images", "train2014")
            self.split_dir = osp.join(self.data_root, "splits")

        if not self.exists_dataset():
            print("The dataset {} is not found!".format(osp.join(self.split_root, self.dataset)))
            print(
                "Please download index cache to data folder: \n \
                https://drive.google.com/open?id=1cZI562MABLtAzM6YU4WmKPFFguuVr0lZ"
            )
            exit(0)

        dataset_path = osp.join(self.split_root, self.dataset)
        valid_splits = self.SUPPORTED_DATASETS[self.dataset]["splits"]

        if self.lstm:
            self.corpus = Corpus()
            corpus_path = osp.join(dataset_path, "corpus.pth")
            self.corpus = torch.load(corpus_path)

        if split not in valid_splits:
            raise ValueError("Dataset {0} does not have split {1}".format(self.dataset, split))

        splits = [split]
        if self.dataset != "referit":
            splits = ["train", "val"] if split == "trainval" else [split]
        for split in splits:
            imgset_file = "{0}_{1}.pth".format(self.dataset, split)
            imgset_path = osp.join(dataset_path, imgset_file)
            self.images += torch.load(imgset_path)

        if self.prompt_template:
            self.images = self.prompt(self.images)

    def exists_dataset(self):
        return osp.exists(osp.join(self.split_root, self.dataset))

    # TODO: 这句是关键
    def pull_item(self, idx):
        if self.dataset == "flickr":
            img_file, bbox, phrase = self.images[idx]
        else:
            img_file, _, bbox, phrase, attri = self.images[idx]
        ## box format: to x1y1x2y2
        bbox_ori = bbox
        if not (self.dataset == "referit" or self.dataset == "flickr"):
            bbox = np.array(bbox, dtype=int)
            bbox[2], bbox[3] = bbox[0] + bbox[2], bbox[1] + bbox[3]
        else:
            bbox = np.array(bbox, dtype=int)

        img_path = osp.join(self.im_dir, img_file)
        img = Image.open(img_path).convert("RGB")

        bbox = torch.tensor(bbox)
        bbox = bbox.float()
        return img_file, img, phrase, bbox, bbox_ori

    def tokenize_phrase(self, phrase):
        return self.corpus.tokenize(phrase, self.query_len)

    def untokenize_word_vector(self, words):
        return self.corpus.dictionary[words]

    # TODO: 新增
    def prompt(self, sample_list):
        n = len(sample_list)
        new_sample_list = []

        for i in range(n):
            if self.dataset == "flickr":
                tmp_sample = (sample_list[i][0], sample_list[i][1], self.prompt_template.replace("{pseudo_query}", sample_list[i][2]))
            else:
                # print("\nsample_list:\n", sample_list[i])
                #  ('COCO_train2014_000000000839.jpg', '482127.pth', [303.58, 69.03, 293.29, 425.79],
                #  'guy flopping around on the right', [('r1', ['guy']), ('r2', ['none']), ('r3', ['none']),
                #  ('r4', ['none']), ('r5', ['none']), ('r6', ['none']), ('r7', ['none']),
                #  ('r8', ['right', 'flopping', 'around'])])
                tmp_sample = (sample_list[i][0], sample_list[i][1], sample_list[i][2], self.prompt_template.replace("{pseudo_query}", sample_list[i][3]), sample_list[i][4])
            new_sample_list.append(tmp_sample)
        return new_sample_list

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_file, img, phrase, bbox, bbox_ori = self.pull_item(idx)  # bbox  x1y1x2y2
        phrase = phrase.lower()
        input_dict = {"img": img, "box": bbox, "text": phrase}
        input_dict = self.transform(input_dict)
        w, h = img.width, img.height
        img = input_dict["img"]
        img_mask = input_dict["mask"]
        bbox = input_dict["box"]
        phrase = input_dict["text"]

        txt_encoded = self.tokenizer([phrase], max_length=self.query_len, padding="max_length", return_tensors="pt")
        text_ids = txt_encoded["input_ids"]
        text_masks = txt_encoded["attention_mask"]
        return {"img": img, "img_mask": img_mask, "text_ids": text_ids, "text_masks": text_masks, "bbox": bbox}
        # return img, np.array(img_mask), np.array(text_ids, dtype=int), np.array(text_masks, dtype=int), \
        #            np.array(bbox, dtype=np.float32), img_file, phrase, bbox_ori, w, h
