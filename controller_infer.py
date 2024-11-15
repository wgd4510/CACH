import os
import shutil
import random
import torch
from models.CACH import CACH_Transformer as Model
from utils.utils import calc_hamming_dist
from torchinfo import summary


class Dataset_img_txt(torch.utils.data.Dataset):
    def __init__(self, images, captions, imgid):
        super().__init__()
        self.images = images
        self.captions = captions
        self.imgid = imgid

    def __getitem__(self, index):
        return (index, self.images[index], self.captions[index], self.imgid[index])

    def __len__(self):
        return len(self.images)


class Controller:
    def __init__(self, log, cfg, dataloaders, data_handler):
        self.logger = log
        self.cfg = cfg
        self.path = cfg.savepath
        self.device = torch.device(self.cfg.cuda_device if torch.cuda.is_available() else "cpu")
        
        self.model = Model(self.cfg)
        self.dataloader_db = dataloaders
        self.data_handler = data_handler
        self.idst = [0, 1, 2, 3, 4, 9, 19]  # 挑选前5，第10，第20的结果

    def load_model(self, weights):
        self.model.load(weights)

    def infer_once(self):
        self.model.to(self.device)
        self.model.eval()

        summary(self.model.img_backbone, input_size=[(1, 3, self.cfg.imgsize, self.cfg.imgsize)], device="cuda")
        summary(self.model.image_module, input_size=[(1, self.cfg.image_dim)], device="cuda")
        # summary(self.model.txt_encoder, input_size=[(2, 64)], device="cuda")
        summary(self.model.text_module, input_size=[(1, 768)], device="cuda")


    def infer(self):
        self.model.to(self.device).train()
        self.model.eval()
        
        rBX, rBY = self.generate_codes_from_dataloader(self.dataloader_db)
        choose_ids = random.sample(range(0, len(self.data_handler.file_names)), 20)

        for choose_id in choose_ids:
            # 随机选择一个查询数据，在数据库中检索对应的结果
            self.infer_each(choose_id, rBX, rBY)
            
    def infer_each(self, choose_id, rBX, rBY):
        img_src = self.data_handler.file_names[choose_id]
        txt_src = self.data_handler.captions[choose_id]
        self.logger.info(f'query image filename: {img_src}')
        self.logger.info(f'query caption: {txt_src}')

        savepath = os.path.join(self.path, img_src.split('.')[0])
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        f_txt = open(os.path.join(savepath, 'caption.txt'), 'w')
        f_txt.write('query caption: {} \n'.format(txt_src))
        img_path = os.path.join(self.cfg.dataset_image_folder_path, img_src)
        shutil.copy(img_path, savepath)

        qBX = rBX[choose_id, :]  # 查询数据的图像Hash码
        qBY = rBY[choose_id, :]  # 查询数据的文本Hash码

        # # 从查询库中随机选一张图片，检索对应的文本
        hamm = calc_hamming_dist(qBX, rBY) 
        _, ind = torch.sort(hamm)  # 从小到大进行排序，并获取对应的序号，距离越小越相似
        ind.squeeze_()
        for i in self.idst:
            caption = self.data_handler.captions[ind[i]]
            self.logger.info(f'{i+1}st caption: {caption}')
            f_txt.write('{:0>2}st caption: {} \n'.format(i+1, caption))

        # # 从查询库中随机选一段文本，检索对应的图片
        hamm = calc_hamming_dist(qBY, rBX)  
        _, ind = torch.sort(hamm)
        ind.squeeze_()
        for i in self.idst:
            file_name = self.data_handler.file_names[ind[i]]
            self.logger.info(f'{i+1}st image file_name: {file_name}')
            f_txt.write('{:0>2}st filename: {} \n'.format(i+1, file_name))
            img_path = os.path.join(self.cfg.dataset_image_folder_path, file_name)
            shutil.copy(img_path, savepath)

    # 计算数据库中所有数据的Hashcode
    def generate_codes_from_dataloader(self, dataloader):
        num = len(dataloader.dataset)
        Bi = torch.zeros(num, self.cfg.hash_dim).to(self.device)
        Bt = torch.zeros(num, self.cfg.hash_dim).to(self.device)

        for i, (idx, img, txt, idimg) in enumerate(dataloader):
            img = img.to(self.device)
            txt = txt.to(self.device) 
            bi = self.model.generate_img_code(img)
            bt = self.model.generate_txt_code(txt)
            bi = bi[:, -self.cfg.hash_dim:]
            bt = bt[:, -self.cfg.hash_dim:]
            idx_end = min(num, (i + 1) * self.cfg.batch_size)
            Bi[i * self.cfg.batch_size:idx_end, :] = bi.data
            Bt[i * self.cfg.batch_size:idx_end, :] = bt.data

        Bi = torch.sign(Bi)
        Bt = torch.sign(Bt)
        return Bi, Bt
