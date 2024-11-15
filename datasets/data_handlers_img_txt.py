import os
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import torchvision
import transformers
from utils.utils import read_json, get_labels, get_image_file_names, get_captions
from configs.config import cfg


class DataHandler_Train:
    def __init__(self):
        super().__init__()
        self.imgaug = True
        self.txtaug = True
        if cfg.img_aug == 'None':
            self.imgaug = False
        if cfg.txt_aug == 'None':
            self.txtaug = False

    def load_train_query_db_data(self):
        """
        Load and split (train, query, db)

        :return: tuples of (images, captions, labels), each element is array
        """
        random.seed(cfg.seed)
        images, captions, labels, captions_aug, images_aug = load_dataset(
            img_aug = self.imgaug, txt_aug = self.txtaug)

        train, query, db = self.split_data(images, captions, labels, captions_aug, images_aug)
        return train, query, db
    
    def split_data(self, images, captions, labels, captions_aug, images_aug):
        """
        Split dataset to get training, query and db subsets

        :param: images: image embeddings array
        :param: captions: caption embeddings array
        :param: labels: labels array
        :param: captions_aug: augmented caption embeddings
        :param: images_aug: augmented image embeddings

        :return: tuples of (images, captions, labels), each element is array
        """
        if cfg.dataset in ['ucm', 'rsicd']:
            idx_tr, idx_q, idx_db = get_split_idxs(len(images))
        elif cfg.dataset in ['sydney']:
            idx_tr, idx_q, idx_db = get_split_idxs_sydeny()

        idx_tr_cap, idx_q_cap, idx_db_cap = get_caption_idxs(idx_tr, idx_q, idx_db)

        if self.imgaug:
            train_img = images_aug[idx_tr]
            query_img = images_aug[idx_q]
            db_img = images_aug[idx_db]
        else:
            train_img = None
            query_img = None
            db_img = None

        if self.txtaug:
            train_txt = captions_aug[idx_tr_cap]
            query_txt = captions_aug[idx_q_cap]
            db_txt = captions_aug[idx_db_cap]
        else:
            train_txt = None
            query_txt = None
            db_txt = None

        train = images[idx_tr], captions[idx_tr_cap], labels[idx_tr], (
            idx_tr, idx_tr_cap), train_txt, train_img
        query = images[idx_q], captions[idx_q_cap], labels[idx_q], (
            idx_q, idx_q_cap), query_txt, query_img
        db = images[idx_db], captions[idx_db_cap], labels[idx_db], (
            idx_db, idx_db_cap), db_txt, db_img
        
        return train, query, db


class DataHandler_Infer:
    def __init__(self):
        super().__init__()
        self.file_names = []
        self.captions = []

    def load_database(self):
        data = read_json(cfg.dataset_json_file)

        # 获取图像路径列表和文本列表
        for img in data['images']:
            self.file_names.append(img['filename'])
            self.captions.append(img['sentences'][random.randint(0, 4)]['raw'])

        images_folder = cfg.dataset_image_folder_path
        img_data = []  # 图像数据
        id_data = []
        img_preprocess = init_transforms(aug_mode='center_crop_only')
        for i, img_name in enumerate(tqdm(self.file_names)):
            img = Image.open(os.path.join(images_folder, img_name))
            img_src = img_preprocess[0](img)
            img_data.append(img_src)
            id_data.append(i)
        output_img = torch.stack(img_data)
        output_id = torch.stack([torch.tensor(id_data)]).reshape(-1, 1)
        print('=========> Load image Data: ', output_img.shape)

        txt_data = []  # 文本数据
        # load tokenizer                       
        tokenizer = transformers.AutoTokenizer.from_pretrained(cfg.txt_weights)
        if cfg.txt_backbone == 'gpt2':
            tokenizer.pad_token = tokenizer.eos_token
        max_token_length = cfg.caption_token_length
        captions_list_t = tqdm(list(range(len(self.captions))))
        for i in captions_list_t:
            item = tokenizer.encode_plus(
                self.captions[i],
                max_length=max_token_length,
                padding='max_length',
                add_special_tokens=True,
                return_attention_mask=True,
                return_tensors='pt')
            txt_data.append(torch.stack((item['input_ids'].squeeze(), item['attention_mask'].squeeze()), 0))
        output_txt_data = torch.stack(txt_data)
        print('=========> Load Caption Data: ', output_txt_data.shape)

        database = output_img, output_txt_data, output_id
        return database

    

def load_dataset(img_aug=False, txt_aug=False):
    """
    Load dataset

    :return: images and captions embeddings, labels
    """
    # read captions from JSON file
    data = read_json(cfg.dataset_json_file)
    labels = np.array(get_labels(data, True))  # 2100
    print('=========> Load Labels Data: ', labels.shape)
    txt_src_data, txt_aug_data = load_txt_txtaug(data, txt_aug)
    img_src_data, img_aug_data = load_img_imgaug(data, img_aug)        
    return img_src_data, txt_src_data, labels, txt_aug_data, img_aug_data


def load_img_imgaug(data, img_aug=False):
    img_data = []
    img_aug_data = []

    # get file names
    file_names = get_image_file_names(data)  # 2100
    images_folder = cfg.dataset_image_folder_path
    img_preprocess = init_transforms(aug_mode='center_crop_only')
    file_names_t = tqdm(file_names)
    for img_name in file_names_t:
        img = Image.open(os.path.join(images_folder, img_name))
        img_src = img_preprocess[0](img)
        img_data.append(img_src)
        if img_aug:
            img_transforms = init_transforms(aug_mode=cfg.img_aug)
            aug_transform = random.choice(img_transforms)
            img_aug_tran = aug_transform(img)
            img_aug_data.append(img_aug_tran)

    output_img = torch.stack(img_data)
    print('=========> Load image Data: ', output_img.shape)
    output_img_aug = None
    if img_aug:
        output_img_aug = torch.stack(img_aug_data)
        print('=========> Load image_Aug Data: ', output_img_aug.shape)       
    return output_img, output_img_aug


def init_transforms(aug_mode):
    """
    Initialize transforms.

    :return: list of transforms sequences (may be only one), one will be selected and applied to image
    """
    if aug_mode == 'each_img_random':
        return init_transforms_random()
    else:
        return [init_transforms_not_random(cfg.img_transforms_dicts[aug_mode])]


def init_transforms_random():
    """
    Initialize transforms randomly from transforms dictionary.

    :return: list of transforms sequences, one will be selected and applied to image
    """
    transforms = []
    for ts in cfg.img_transforms_dicts[cfg.img_aug]:
        transforms.append(init_transforms_not_random(cfg.img_transforms_dicts[ts]))
    return transforms


def init_transforms_not_random(transform_dict):
    """
    Initialize transforms non-randomly from transforms dictionary.
    :param transform_dict: transforms dictionary from config file

    :return: sequence of transforms to apply to each image
    """

    def _rotation_transform(values):
        return torchvision.transforms.RandomChoice(
            [torchvision.transforms.RandomRotation(val) for val in values])

    def _affine_transform(values):
        return torchvision.transforms.RandomChoice(
            [torchvision.transforms.RandomAffine(val) for val in values])

    def _gaussian_blur_transform(values):
        return torchvision.transforms.GaussianBlur(*values)

    def _center_crop_transform(values):
        return torchvision.transforms.CenterCrop(values)

    def _random_crop_transform(values):
        return torchvision.transforms.RandomCrop(values)

    def _color_jittering(values):
        return torchvision.transforms.ColorJitter(0.8 * values, 0.8 * values,
                                                  0.8 * values, 0.2 * values)

    image_transform_funcs = {
        'rotation': _rotation_transform,
        'affine': _affine_transform,
        'blur': _gaussian_blur_transform,
        'center_crop': _center_crop_transform,
        'random_crop': _random_crop_transform,
        'jitter': _color_jittering
    }

    transforms_list = []
    for k, v in transform_dict.items():
        transforms_list.append(image_transform_funcs[k](v))

    transforms_list.append(torchvision.transforms.Resize(cfg.imgsize))
    transforms_list.append(torchvision.transforms.ToTensor())
    return torchvision.transforms.Compose(transforms_list)


def load_txt_txtaug(data, txt_aug=False):
    txt_data = []
    txt_aug_data = []
    # get captions
    captions, aug_captions_rb, aug_captions_bt_prob, aug_captions_bt_chain = get_captions(
        data)
    # load tokenizer                       
    tokenizer = transformers.AutoTokenizer.from_pretrained(cfg.txt_weights)
    if cfg.txt_backbone == 'gpt2':
        tokenizer.pad_token = tokenizer.eos_token

    # max_token_length = get_max_token_length(tokenizer, captions)
    max_token_length = cfg.caption_token_length
    if txt_aug:
        if cfg.txt_aug == 'default' or cfg.txt_aug == 'rule-based':
            caption_aug = aug_captions_rb
        elif cfg.txt_aug == 'backtranslation-prob':
            caption_aug = aug_captions_bt_prob
        elif cfg.txt_aug == 'backtranslation-chain':
            caption_aug = aug_captions_bt_chain
    captions_list_t = tqdm(list(range(len(captions))))
    for i in captions_list_t:
        # for i in range(len(captions)):
        item = tokenizer.encode_plus(
            captions[i],
            max_length=max_token_length,
            padding='max_length',
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors='pt')
        txt_data.append(
            torch.stack((item['input_ids'].squeeze(),
                         item['attention_mask'].squeeze()), 0))
        if txt_aug:
            item_aug = tokenizer.encode_plus(
                caption_aug[i],
                max_length=max_token_length,
                padding='max_length',
                add_special_tokens=True,
                return_attention_mask=True,
                return_tensors='pt')
            txt_aug_data.append(
                torch.stack((item_aug['input_ids'].squeeze(),
                             item_aug['attention_mask'].squeeze()), 0))

    output_txt_data = torch.stack(txt_data)
    print('=========> Load Caption Data: ', output_txt_data.shape)
    output_txt_aug_data = None
    if txt_aug:
        output_txt_aug_data = torch.stack(txt_aug_data)
        print('=========> Load CaptionAug Data: ', output_txt_aug_data.shape)
    return output_txt_data, output_txt_aug_data


def get_split_idxs(arr_len):
    """
    Get indexes for training, query and db subsets

    :param: arr_len: array length

    :return: indexes for training, query and db subsets
    """
    idx_all = list(range(arr_len))
    idx_train, idx_eval = split_indexes(idx_all, cfg.dataset_train_split)
    idx_query, idx_db = split_indexes(idx_eval, cfg.dataset_query_split)
    return idx_train, idx_query, idx_db


def split_indexes(idx_all, split):
    """
    Splits list in two parts.

    :param idx_all: array to split
    :param split: portion to split
    :return: splitted lists
    """
    idx_length = len(idx_all)
    selection_length = int(idx_length * split)
    idx_selection = sorted(random.sample(idx_all, selection_length))
    idx_rest = sorted(list(set(idx_all).difference(set(idx_selection))))
    return idx_selection, idx_rest


def get_split_idxs_sydeny():
    idx_train, idx_query, idx_db = [], [], []
    data = read_json(cfg.dataset_json_file)
    for img in data['images']:
        imgid = img['imgid']
        split = img['split']
        if split == 'train':
            idx_train.append(imgid)
        elif split == 'val':
            idx_query.append(imgid)
        elif split == 'test':
            idx_db.append(imgid)
    return idx_train, idx_query, idx_db


def get_caption_idxs(idx_train, idx_query, idx_db):
    """
    Get caption indexes.

    :param: idx_train: train image (and label) indexes
    :param: idx_query: query image (and label) indexes
    :param: idx_db: db image (and label) indexes

    :return: caption indexes for corresponding index sets
    """
    idx_train_cap = get_caption_idxs_from_img_idxs(idx_train)
    idx_query_cap = get_caption_idxs_from_img_idxs(idx_query)
    idx_db_cap = get_caption_idxs_from_img_idxs(idx_db)
    return idx_train_cap, idx_query_cap, idx_db_cap


def get_caption_idxs_from_img_idxs(img_idxs):
    """
    Get caption indexes. There are 5 captions for each image (and label).
    Say, img indexes - [0, 10, 100]
    Then, caption indexes - [0, 1, 2, 3, 4, 50, 51, 52, 53, 54, 100, 501, 502, 503, 504]

    :param: img_idxs: image (and label) indexes

    :return: caption indexes
    """
    caption_idxs = []
    for idx in img_idxs:
        for i in range(5):  # each image has 5 captions
            caption_idxs.append(idx * 5 + i)
    return caption_idxs


def get_dataloaders(data_handler, ds_train, ds_query, ds_db):
    """
    Initializes dataloaders

    :param data_handler: data handler instance
    :param ds_train: class of train dataset
    :param ds_query: class of query dataset
    :param ds_db: class of database dataset

    :return: dataloaders
    """
    data_handler = data_handler()
    # 加载数据 并进行数据划分，划分后数据格式：data tuples:
    # (images, captions, labels, (idxs, idxs_cap)) or (images, captions, labels, (idxs, idxs_cap), augmented_captions)
    # 默认是：images[idx_tr], captions[idx_tr_cap], labels[idx_tr], (idx_tr, idx_tr_cap), captions_aug[idx_tr_cap]
    # train_tuple: 1050; query_tuple: 210; db_tuple: 840
    train_tuple, query_tuple, db_tuple = data_handler.load_train_query_db_data()

    # train dataloader
    dataset_triplets = ds_train(*train_tuple, seed=cfg.seed)
    dataloader_train = DataLoader(
        dataset_triplets, batch_size=cfg.batch_size, shuffle=True)

    # query dataloader
    dataset_q = ds_query(*query_tuple, seed=cfg.seed)
    dataloader_q = DataLoader(dataset_q, batch_size=cfg.batch_size)

    # database dataloader
    dataset_db = ds_db(*db_tuple, seed=cfg.seed)
    dataloader_db = DataLoader(dataset_db, batch_size=cfg.batch_size)

    return dataloader_train, dataloader_q, dataloader_db
