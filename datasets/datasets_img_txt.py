import torch
import numpy as np
from utils.utils import select_idxs


class AbstractDataset(torch.utils.data.Dataset):
    def __init__(self,
                 images,
                 captions,
                 labels,
                 idxs,
                 captions_aug=None,
                 images_aug=None,
                 seed=42):
        self.seed = seed
        self.image_replication_factor = 1  # default value, how many times we need to replicate image

        self.images = images
        self.captions = captions
        self.labels = labels
        self.captions_aug = captions_aug
        self.images_aug = images_aug

        self.idxs = np.array(idxs[0])
        self.idxs_cap = np.array(idxs[1])

    def __getitem__(self, index):
        return

    def __len__(self):
        return
    
    
class Dataset_img_img_txt_txt(AbstractDataset):
    def __init__(self,
                 images,
                 captions,
                 labels,
                 idxs,
                 captions_aug=None,
                 images_aug=None,
                 seed=42):
        super().__init__(images, captions, labels, idxs, captions_aug, images_aug, seed)
        per_img_captions = int(len(self.captions) / len(self.images))
        caption_idxs = select_idxs(len(self.captions), 1, 5, seed=self.seed)[0]
        self.idxs_cap = self.idxs_cap[caption_idxs]

        self.captions = self.captions[caption_idxs]
        self.captions_aug = self.captions_aug[caption_idxs]
        print('=========> Datasets captions: ', self.captions.shape)
        print('=========> Datasets captions_aug: ', self.captions_aug.shape)

    def __getitem__(self, index):
        """
        Returns a tuple (img, txt1, txt2, label) - image and 2 corresponding captions

        :param index: index of sample
        :return: tuple (img, txt1, txt2, label)
        """
        return (index, (self.idxs[index], self.idxs[index],
                        self.idxs_cap[index],
                        self.idxs_cap[index]), self.images[index],
                self.images_aug[index], self.captions[index],
                self.captions_aug[index], self.labels[index])

    def __len__(self):
        return len(self.images)


class Dataset_img_txt(AbstractDataset):
    """
    Class for dataset representation.

    Each image has 5 corresponding captions

    Duplet dataset sample - img-txt (image and corresponding caption)
    """

    def __init__(self,
                 images,
                 captions,
                 labels,
                 idxs,
                 captions_aug=None,
                 images_aug=None,
                 seed=42):
        """
        Initialization.

        :param images: image embeddings vector
        :param captions: captions embeddings vector
        :param labels: labels vector
        """
        super().__init__(images, captions, labels, idxs, captions_aug, images_aug, seed)
        per_img_captions = int(len(self.captions) / len(self.images))  # 5
        # caption 的长度是imgs的五倍，所以select_idxs 将 self.captions每组5个随机抽取1个放到新的self.captions
        caption_idxs = select_idxs(len(self.captions), 1, 5, seed=self.seed)[0]
        self.idxs_cap = self.idxs_cap[caption_idxs]
        self.captions = self.captions[caption_idxs]
        print('=========> Datasets captions: ', self.captions.shape)

    def __getitem__(self, index):
        """
        Returns a tuple (img, txt, label) - image and corresponding caption

        :param index: index of sample
        :return: tuple (img, txt, label)
        """
        return (index, (self.idxs[index], self.idxs_cap[index]),
                self.images[index], self.captions[index], self.labels[index])

    def __len__(self):
        return len(self.images)
