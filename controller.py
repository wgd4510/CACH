import os
import time
import random
import torch
from torch.optim import Adam
from torch.nn.functional import one_hot
from models.CACH import CACH_Transformer as Model
from utils.contrastive_loss import NTXentLoss
from utils.utils import write_pickle, calc_map_k, calc_map_rad, pr_curve, p_top_k, calc_hamming_dist


class Controller:
    def __init__(self, log, cfg, dataloaders):
        self.since = time.time()
        self.logger = log
        self.cfg = cfg
        self.path = cfg.savepath
        self.device = torch.device(
            self.cfg.cuda_device if torch.cuda.is_available() else "cpu")
        
        self.data_train, self.dataloader_q, self.dataloader_db = dataloaders

        self.B, self.Hi1, self.Hi2, self.Ht1, self.Ht2 = self.init_hashes()
        self.losses = []

        self.maps_max = {'i2t': 0., 't2i': 0., 'i2i': 0., 't2t': 0., 'avg': 0.}
        self.maps = {'i2t': [], 't2i': [], 'i2i': [], 't2t': [], 'avg': []}

        self.model = Model(self.cfg)

        self.f_id = self.get_feature_ids()
        self.optimizer_gen, self.optimizer_dis = self.get_optimizers()

        self.contr_loss = NTXentLoss()
        self.cos_loss = torch.nn.CosineEmbeddingLoss(reduction='mean')

    def init_hashes(self):
        dataset_size = len(self.data_train.dataset)

        B = torch.randn(dataset_size, self.cfg.hash_dim).sign().to(self.device)  # [1050, bit]  [-1, 1]
        Hi1 = torch.zeros(dataset_size, self.cfg.hash_dim).sign().to(self.device)
        Hi2 = torch.zeros(dataset_size, self.cfg.hash_dim).sign().to(self.device)
        Ht1 = torch.zeros(dataset_size, self.cfg.hash_dim).sign().to(self.device)
        Ht2 = torch.zeros(dataset_size, self.cfg.hash_dim).sign().to(self.device)
        return B, Hi1, Hi2, Ht1, Ht2
    
    def get_optimizers(self):
        optimizer_gen = Adam(
            [{'params': self.model.img_backbone.parameters()}, 
                {'params': self.model.image_module.parameters()}, 
                {'params': self.model.txt_encoder.parameters()}, 
                {'params': self.model.text_module.parameters()}],
            lr=self.cfg.lr,
            weight_decay=0.0005)

        optimizer_dis = Adam(
            [{'params': self.model.img_decoder_i2i.parameters()}, 
                {'params': self.model.img_decoder_i2t.parameters()},
                {'params': self.model.text_decoder_t2t.parameters()},
                {'params': self.model.text_decoder_t2i.parameters()}],
            lr=self.cfg.lr,
            betas=(0.5, 0.9),
            weight_decay=0.0001)
        return optimizer_gen, optimizer_dis
    
    def get_feature_ids(self):
        f_id = [0]
        f = self.cfg.hidden_dim
        for i in range(4):
            f_id.append(f_id[-1] + f)
            f = f // 2
        f_id.append(f_id[-1] + self.cfg.hash_dim)
        return f_id
    
    def train_epoch(self, epoch):
        t1 = time.time()
        self.model.to(self.device).train()

        for i, (idx, sample_idxs, img1, img2, txt1, txt2, label) in enumerate(self.data_train):
            img1 = img1.to(self.device)  # original images
            img2 = img2.to(self.device)  # augmented images
            txt1 = txt1.to(self.device)  # original texts
            txt2 = txt2.to(self.device)  # augmented texts

            i_src, i_aug, t_src, t_aug = self.model(img1, img2, txt1, txt2)

            self.Hi1[idx, :] = i_src[:, self.f_id[4]: self.f_id[5]]
            self.Hi2[idx, :] = i_aug[:, self.f_id[4]: self.f_id[5]]
            self.Ht1[idx, :] = t_src[:, self.f_id[4]: self.f_id[5]]
            self.Ht2[idx, :] = t_aug[:, self.f_id[4]: self.f_id[5]]
            # cross auto-encode reconstructed loss: I->I, I->T, T->T, T->T
            if self.cfg.decoder_loss:
                decoder_loss = self.train_decoder(i_src, i_aug, t_src, t_aug)
            else:
                decoder_loss = 0
            # Semantic similarity loss and Contrastive loss
            encoder_loss = self.train_encoder(i_src, i_aug, t_src, t_aug)

        self.B = (((self.Hi1.detach() + self.Hi2.detach()) / 2 +
                   (self.Ht1.detach() + self.Ht2.detach()) / 2) / 2).sign()
        
        self.update_optimizer_params(epoch)
        
        delta_t = time.time() - t1
        s = '[{}/{}] Train: {:.3f}s, Losses: Decoder_loss: {:.2f}, Encoder_loss: {:.2f}'
        self.logger.info(s.format(epoch + 1, self.cfg.epochs, delta_t, decoder_loss, encoder_loss))
        
    # train encoder
    def train_encoder(self, i_src, i_aug, t_src, t_aug):
        img_hash_src = i_src[:, self.f_id[4]: self.f_id[5]]
        img_hash_aug = i_aug[:, self.f_id[4]: self.f_id[5]]
        txt_hash_src = t_src[:, self.f_id[4]: self.f_id[5]]
        txt_hash_aug = t_aug[:, self.f_id[4]: self.f_id[5]]

        # Semantic similarity loss: calculate feature graphs Cosine similarity loss 
        if self.cfg.feature_loss:
            loss_feature_cosine = self.calc_feature_loss(i_src, i_aug, t_src, t_aug)
        else:
            loss_feature_cosine = 0
        # Contrastive loss: calculate NTXentLoss(img、img_aug、text、text_aug) 
        loss_ntxent = self.calc_ntxent_loss(img_hash_src, img_hash_aug, txt_hash_src, txt_hash_aug)

        loss_encoder = 0.001 * loss_feature_cosine + loss_ntxent
        # loss_encoder = loss_ntxent
        self.optimizer_gen.zero_grad()
        loss_encoder.backward()
        self.optimizer_gen.step()
        return loss_encoder
    
    def calc_feature_loss(self, i_src, i_aug, t_src, t_aug):
        loss_flag = torch.ones(i_src.shape[0]).to(self.device)
        # img and text   Cosine similarity loss
        loss_is_ts = self.calc_cosine_loss(i_src, t_src, loss_flag) * self.cfg.encoder_weights[0]
        # img_aug and text_aug   Cosine similarity loss
        loss_ia_ta = self.calc_cosine_loss(i_aug, t_aug, loss_flag) * self.cfg.encoder_weights[1]
        # img and text_aug   Cosine similarity loss
        loss_is_ta = self.calc_cosine_loss(i_src, t_aug, loss_flag) * self.cfg.encoder_weights[2]
        # img_aug and text   Cosine similarity loss
        loss_ia_ts = self.calc_cosine_loss(i_aug, t_src, loss_flag) * self.cfg.encoder_weights[3]
        return loss_is_ts + loss_ia_ta + loss_is_ta + loss_ia_ts

    # calculate img and text  Cosine similarity loss
    def calc_cosine_loss(self, img, text, loss_flag):
        img_1 = img[:, self.f_id[1]: self.f_id[2]]
        img_2 = img[:, self.f_id[2]: self.f_id[3]]
        img_3 = img[:, self.f_id[3]: self.f_id[4]]
        txt_1 = text[:, self.f_id[1]: self.f_id[2]]
        txt_2 = text[:, self.f_id[2]: self.f_id[3]]
        txt_3 = text[:, self.f_id[3]: self.f_id[4]]

        f1_loss = self.cos_loss(img_1, txt_1, loss_flag) * self.cfg.feature_weights[0]
        f2_loss = self.cos_loss(img_2, txt_2, loss_flag) * self.cfg.feature_weights[1]
        f3_loss = self.cos_loss(img_3, txt_3, loss_flag) * self.cfg.feature_weights[2]

        e_loss = f1_loss + f2_loss + f3_loss
        return e_loss
    
    # calculate NTXentLoss
    def calc_ntxent_loss(self, h_img1, h_img2, h_txt1, h_txt2):
        loss_ntxent_inter = self.contr_loss(
            h_img1, h_txt1, type='orig') * float(self.cfg.contrastive_weights[0])
        loss_ntxent_intra_img = self.contr_loss(
            h_img1, h_img2, type='orig') * float(self.cfg.contrastive_weights[1])
        loss_ntxent_intra_txt = self.contr_loss(
            h_txt1, h_txt2, type='orig') * float(self.cfg.contrastive_weights[2])
        loss_ntxent = loss_ntxent_inter + loss_ntxent_intra_txt + loss_ntxent_intra_img
        return loss_ntxent
    
    # train decoder
    def train_decoder(self, i_src, i_aug, t_src, t_aug):
        self.optimizer_dis.zero_grad()
        decoder_loss = self.train_decoder_step(i_src, t_src)
        decoder_aug_loss = self.train_decoder_step(i_aug, t_aug)
        self.optimizer_dis.step()
        return decoder_loss + decoder_aug_loss
    
    # Reconstruction feature calculate cosine similarity loss
    def train_decoder_step(self, img, text):
        img_f = img[:, self.f_id[0]: self.f_id[1]]
        img_f3 = img[:, self.f_id[3]: self.f_id[4]]
        txt_f = text[:, self.f_id[0]: self.f_id[1]]
        txt_f3 = text[:, self.f_id[3]: self.f_id[4]]
        loss_flag = torch.ones(img_f.shape[0]).to(self.device)
        f3_i2i = self.model.img_decoder_i2i(img_f3)
        f3_i2t = self.model.img_decoder_i2t(img_f3)
        f3_t2t = self.model.text_decoder_t2t(txt_f3)
        f3_t2i = self.model.text_decoder_t2i(txt_f3)

        d_i2i = self.cos_loss(img_f, f3_i2i, loss_flag)
        d_t2i = self.cos_loss(img_f, f3_t2i, loss_flag)
        d_i2t = self.cos_loss(txt_f, f3_i2t, loss_flag)
        d_t2t = self.cos_loss(txt_f, f3_t2t, loss_flag)

        decoder_loss = d_i2i + d_t2i + d_i2t + d_t2t
        decoder_loss.backward(retain_graph=True)
        return decoder_loss

    def update_optimizer_params(self, epoch):
        if epoch % 50 == 0:
            for params in self.optimizer_gen.param_groups:
                params['lr'] = max(params['lr'] * 0.8, 1e-6)

    def eval(self, epoch):
        """
        Evaluate model. Calculate MAPs for current epoch
        Save model and hashes if current epoch is the best

        :param: epoch: current epoch
        """
        self.model.eval()

        qBX, qBY, rBX, rBY, qLX, qLY, rLX, rLY, indexes = self.get_codes_labels_indexes()

        mapi2t, mapt2i, mapi2i, mapt2t, mapavg = self.calc_maps_k(
            qBX, qBY, rBX, rBY, qLX, qLY, rLX, rLY, self.cfg.retrieval_map_k)

        map_k_5 = (mapi2t, mapt2i, mapi2i, mapt2t, mapavg)
        map_k_10 = self.calc_maps_k(qBX, qBY, rBX, rBY, qLX, qLY, rLX, rLY, 10)
        map_k_20 = self.calc_maps_k(qBX, qBY, rBX, rBY, qLX, qLY, rLX, rLY, 20)
        rs = range(min(self.cfg.hash_dim, 6))
        map_r = self.calc_maps_rad(qBX, qBY, rBX, rBY, qLX, qLY, rLX, rLY, rs)
        p_at_k = self.calc_p_top_k(qBX, qBY, rBX, rBY, qLX, qLY, rLX, rLY)
        maps_eval = (map_k_5, map_k_10, map_k_20, map_r, p_at_k)

        self.update_maps_dict(mapi2t, mapt2i, mapi2i, mapt2t, mapavg)

        if mapavg > self.maps_max['avg']:
            self.update_max_maps_dict(mapi2t, mapt2i, mapi2i, mapt2t, mapavg)

            self.save_model()
            self.save_hash_codes()

        self.save_model('last')
        write_pickle(os.path.join(self.path, 'maps_eval.pkl'), maps_eval)

        self.model.train()

    def calc_maps_k(self, qBX, qBY, rBX, rBY, qLX, qLY, rLX, rLY, k):
        """
        Calculate MAPs, in regards to K

        :param: qBX: query hashes, modality X
        :param: qBY: query hashes, modality Y
        :param: rBX: response hashes, modality X
        :param: rBY: response hashes, modality Y
        :param: qLX: query labels, modality X
        :param: qLY: query labels, modality Y
        :param: rLX: response labels, modality X
        :param: rLY: response labels, modality Y
        :param: k: k

        :returns: MAPs
        """
        mapi2t = calc_map_k(qBX, rBY, qLX, rLY, k)
        mapt2i = calc_map_k(qBY, rBX, qLY, rLX, k)
        mapi2i = calc_map_k(qBX, rBX, qLX, rLX, k)
        mapt2t = calc_map_k(qBY, rBY, qLY, rLY, k)

        avg = (mapi2t.item() + mapt2i.item() + mapi2i.item() + mapt2t.item()) * 0.25

        mapi2t, mapt2i, mapi2i, mapt2t = mapi2t.item(), mapt2i.item(), mapi2i.item(), mapt2t.item()
        mapavg = avg

        s = 'Valid: mAP@{:2d}, avg: {:3.3f}, i->t: {:3.3f}, t->i: {:3.3f}, i->i: {:3.3f}, t->t: {:3.3f}'
        self.logger.info(s.format(k, mapavg, mapi2t, mapt2i, mapi2i, mapt2t))

        return mapi2t, mapt2i, mapi2i, mapt2t, mapavg

    def calc_maps_rad(self, qBX, qBY, rBX, rBY, qLX, qLY, rLX, rLY, rs):
        """
        Calculate MAPs, in regard to Hamming radius

        :param: qBX: query hashes, modality X
        :param: qBY: query hashes, modality Y
        :param: rBX: response hashes, modality X
        :param: rBY: response hashes, modality Y
        :param: qLX: query labels, modality X
        :param: qLY: query labels, modality Y
        :param: rLX: response labels, modality X
        :param: rLY: response labels, modality Y
        :param: rs: hamming radiuses to output

        :returns: MAPs
        """
        mapsi2t = calc_map_rad(qBX, rBY, qLX, rLY)
        mapst2i = calc_map_rad(qBY, rBX, qLY, rLX)
        mapsi2i = calc_map_rad(qBX, rBX, qLX, rLX)
        mapst2t = calc_map_rad(qBY, rBY, qLY, rLY)

        mapsi2t, mapsi2i = mapsi2t.numpy(), mapsi2i.numpy()
        mapst2i, mapst2t = mapst2i.numpy(), mapst2t.numpy()

        s = 'Valid: mAP HR{}, i->t: {:3.3f}, t->i: {:3.3f}, i->i: {:3.3f}, t->t: {:3.3f}'
        for r in rs:
            self.logger.info(s.format(r, mapsi2t[r], mapst2i[r], mapsi2i[r], mapst2t[r]))

        return mapsi2t, mapst2i, mapsi2i, mapst2t

    def get_codes_labels_indexes(self):
        """
        Generate binary codes from duplet dataloaders for query and response

        :param: remove_replications: remove replications from dataset

        :returns: hash codes and labels for query and response, sample indexes
        """
        # hash X, hash Y, labels X/Y, image replication factor, indexes X, indexes Y
        # qBX图像哈希码，qBY文本哈希码，qLXY标签0-1值，irf_q=1，(qIX, qIY)是图像文本索引
        # 以RSICD-Bit64为例，qBX和qBY是[1092, 64]的-1和1哈希矩阵，rBX和rBY是[4369, 64]的-1和1哈希矩阵，qLXY和rLXY是[1092, 31]的0和1标签矩阵
        # 获取查询库数据的结果
        qBX, qBY, qLXY, irf_q, (qIX, qIY) = self.generate_codes_from_dataloader(self.dataloader_q)
        # 获取数据库数据结果
        rBX, rBY, rLXY, irf_db, (rIX, rIY) = self.generate_codes_from_dataloader(self.dataloader_db)

        # get Y Labels
        qLY = qLXY
        rLY = rLXY

        # X modality sometimes contains replicated samples (see datasets), remove them by selecting each nth element， 因为默认irf_q==irf_db==1，所以下面的操作默认是无效的
        # remove replications for hash codes
        qBX = self.get_each_nth_element(qBX, irf_q)
        rBX = self.get_each_nth_element(rBX, irf_db)
        # remove replications for labels
        qLX = self.get_each_nth_element(qLXY, irf_q)
        rLX = self.get_each_nth_element(rLXY, irf_db)
        # remove replications for indexes
        qIX = self.get_each_nth_element(qIX, irf_q)
        rIX = self.get_each_nth_element(rIX, irf_db)
        # qLX==qLY  rLY==rLX 表示查询词和数据库对应图像和文本的标签值，所以对应的相等
        return qBX, qBY, rBX, rBY, qLX, qLY, rLX, rLY, (qIX, qIY, rIX, rIY)

    @staticmethod
    def get_each_nth_element(arr, n):
        """
        intentionally ugly solution, needed to avoid query replications during test/validation

        :return: array
        """
        return arr[::n]

    def generate_codes_from_dataloader(self, dataloader):
        """
        Generate binary codes from duplet dataloader

        :param: dataloader: duplet dataloader

        :returns: hash codes for given duplet dataloader, image replication factor of dataset
        """
        num = len(dataloader.dataset)

        irf = dataloader.dataset.image_replication_factor

        Bi = torch.zeros(num, self.cfg.hash_dim).to(self.device)
        Bt = torch.zeros(num, self.cfg.hash_dim).to(self.device)
        L = torch.zeros(num, self.cfg.label_dim).to(self.device)

        dataloader_idxs = []

        # for i, input_data in tqdm(enumerate(test_dataloader)):
        for i, (idx, sample_idxs, img, txt, label) in enumerate(dataloader):
            dataloader_idxs = self.stack_idxs(dataloader_idxs, sample_idxs)
            img = img.to(self.device)
            txt = txt.to(self.device)
            if len(label.shape) == 1:
                # label = one_hot(label, num_classes=self.cfg.label_dim).to(self.device)
                label = one_hot(
                    label.to(torch.int64),
                    num_classes=self.cfg.label_dim).to(self.device)
            else:
                label.to(self.device)
            bi = self.model.generate_img_code(img)
            bt = self.model.generate_txt_code(txt)
            bi = bi[:, -self.cfg.hash_dim:]
            bt = bt[:, -self.cfg.hash_dim:]
            idx_end = min(num, (i + 1) * self.cfg.batch_size)
            Bi[i * self.cfg.batch_size:idx_end, :] = bi.data
            Bt[i * self.cfg.batch_size:idx_end, :] = bt.data
            L[i * self.cfg.batch_size:idx_end, :] = label.data

        Bi = torch.sign(Bi)
        Bt = torch.sign(Bt)
        return Bi, Bt, L, irf, dataloader_idxs

    @staticmethod
    def stack_idxs(idxs, idxs_batch):
        if len(idxs) == 0:
            return [ib for ib in idxs_batch]
        else:
            return [torch.hstack(i).detach() for i in zip(idxs, idxs_batch)]

    def update_maps_dict(self, mapi2t, mapt2i, mapi2i, mapt2t, mapavg):
        """
        Update MAPs dictionary (append new values)

        :param: mapi2t: I-to-T MAP
        :param: mapt2i: T-to-I MAP
        :param: mapi2i: I-to-I MAP
        :param: mapt2t: T-to-T MAP
        :param: mapavg: average MAP
        """
        self.maps['i2t'].append(mapi2t)
        self.maps['t2i'].append(mapt2i)
        self.maps['i2i'].append(mapi2i)
        self.maps['t2t'].append(mapt2t)
        self.maps['avg'].append(mapavg)

    def update_max_maps_dict(self, mapi2t, mapt2i, mapi2i, mapt2t, mapavg):
        """
        Update max MAPs dictionary (replace values)

        :param: mapi2t: I-to-T MAP
        :param: mapt2i: T-to-I MAP
        :param: mapi2i: I-to-I MAP
        :param: mapt2t: T-to-T MAP
        :param: mapavg: average MAP
        """
        self.maps_max['i2t'] = mapi2t
        self.maps_max['t2i'] = mapt2i
        self.maps_max['i2i'] = mapi2i
        self.maps_max['t2t'] = mapt2t
        self.maps_max['avg'] = mapavg

    def save_hash_codes(self):
        """
        Save hash codes on a disk
        """
        with torch.cuda.device(self.device):
            torch.save([self.Hi1, self.Hi2, self.Ht1, self.Ht2],
                       os.path.join(self.path, 'hash_codes_i_t.pth'))
        with torch.cuda.device(self.device):
            torch.save(self.B, os.path.join(self.path, 'hash_code.pth'))

    def training_complete(self):
        """
        Output training summary: time and best results
        """
        self.save_train_results_dict()

        current = time.time()
        delta = current - self.since
        self.logger.info('Training complete in {:.0f}m {:.0f}s'.format(delta // 60, delta % 60))
        s = 'Max Avg MAP: {avg:3.3f}, Max MAPs: i->t: {i2t:3.3f}, t->i: {t2i:3.3f}, i->i: {i2i:3.3f}, t->t: {t2t:3.3f}'
        self.logger.info(s.format(**self.maps_max))

    def save_train_results_dict(self):
        """
        Save training history dictionary
        """
        res = self.maps
        res['losses'] = self.losses
        write_pickle(os.path.join(self.path, 'train_res_dict.pkl'), res)

    def test(self):
        """
        Test model. Calculate MAPs, PR-curves and P@K values.
        """
        self.model.to(self.device).train()
        self.model.eval()

        qBX, qBY, rBX, rBY, qLX, qLY, rLX, rLY, indexes = self.get_codes_labels_indexes()

        maps = self.calc_maps_k(qBX, qBY, rBX, rBY, qLX, qLY, rLX, rLY, self.cfg.retrieval_map_k)
        map_dict = self.make_maps_dict(*maps)
        pr_dict = self.calc_pr_curves(qBX, qBY, rBX, rBY, qLX, qLY, rLX, rLY)
        pk_dict = self.calc_p_top_k(qBX, qBY, rBX, rBY, qLX, qLY, rLX, rLY)
        self.save_test_results_dicts(map_dict, pr_dict, pk_dict)

        self.model.train()

        current = time.time()
        delta = current - self.since
        self.logger.info('Test complete in {:.0f}m {:.0f}s'.format(delta // 60, delta % 60))

    def make_maps_dict(self, mapi2t, mapt2i, mapi2i, mapt2t, mapavg):
        """
        Make MAP dict from MAP values

        :param: mapi2t: I-to-T MAP
        :param: mapt2i: T-to-I MAP
        :param: mapi2i: I-to-I MAP
        :param: mapt2t: T-to-T MAP
        :param: mapavg: Average MAP

        :returns: MAPs dictionary
        """

        map_dict = {
            'mapi2t': mapi2t,
            'mapt2i': mapt2i,
            'mapi2i': mapi2i,
            'mapt2t': mapt2t,
            'mapavg': mapavg
        }

        s = 'Avg MAP: {:3.3f}, MAPs: i->t: {:3.3f}, t->i: {:3.3f}, i->i: {:3.3f}, t->t: {:3.3f}'
        self.logger.info(s.format(mapavg, mapi2t, mapt2i, mapi2i, mapt2t))

        return map_dict

    def calc_pr_curves(self, qBX, qBY, rBX, rBY, qLX, qLY, rLX, rLY):
        """
        Calculate PR-curves

        :param: qBX: query hashes, modality X
        :param: qBY: query hashes, modality Y
        :param: rBX: response hashes, modality X
        :param: rBY: response hashes, modality Y
        :param: qLX: query labels, modality X
        :param: qLY: query labels, modality Y
        :param: rLX: response labels, modality X
        :param: rLY: response labels, modality Y

        :returns: PR-curves dictionary
        """
        p_i2t, r_i2t = pr_curve(qBX, rBY, qLX, rLY, tqdm_label='I2T')
        p_t2i, r_t2i = pr_curve(qBY, rBX, qLY, rLX, tqdm_label='T2I')
        p_i2i, r_i2i = pr_curve(qBX, rBX, qLX, rLX, tqdm_label='I2I')
        p_t2t, r_t2t = pr_curve(qBY, rBY, qLY, rLY, tqdm_label='T2T')

        pr_dict = {
            'pi2t': p_i2t,
            'ri2t': r_i2t,
            'pt2i': p_t2i,
            'rt2i': r_t2i,
            'pi2i': p_i2i,
            'ri2i': r_i2i,
            'pt2t': p_t2t,
            'rt2t': r_t2t
        }

        self.logger.info('Precision-recall values: {}'.format(pr_dict))

        return pr_dict

    def calc_p_top_k(self, qBX, qBY, rBX, rBY, qLX, qLY, rLX, rLY):
        """
        Calculate P@K values

        :param: qBX: query hashes, modality X
        :param: qBY: query hashes, modality Y
        :param: rBX: response hashes, modality X
        :param: rBY: response hashes, modality Y
        :param: qLX: query labels, modality X
        :param: qLY: query labels, modality Y
        :param: rLX: response labels, modality X
        :param: rLY: response labels, modality Y

        :returns: P@K values
        """
        k = [1, 5, 10, 20, 50] + list(range(100, 1001, 100))

        pk_i2t = p_top_k(qBX, rBY, qLX, rLY, k, tqdm_label='I2T')
        pk_t2i = p_top_k(qBY, rBX, qLY, rLX, k, tqdm_label='T2I')
        pk_i2i = p_top_k(qBX, rBX, qLX, rLX, k, tqdm_label='I2I')
        pk_t2t = p_top_k(qBY, rBY, qLY, rLY, k, tqdm_label='T2T')

        pk_dict = {
            'k': k,
            'pki2t': pk_i2t,
            'pkt2i': pk_t2i,
            'pki2i': pk_i2i,
            'pkt2t': pk_t2t
        }

        self.logger.info('P@K values: {}'.format(pk_dict))

        return pk_dict

    def save_test_results_dicts(self, map_dict, pr_dict, pk_dict):
        """
        Save test results dictionary

        :param: map_dict: MAPs dictionary
        :param: pr_dict: PR-curves dictionary
        :param: pk_dict: P@K values dictionary
        """
        write_pickle(os.path.join(self.path, 'map_dict.pkl'), map_dict)
        write_pickle(os.path.join(self.path, 'pr_dict.pkl'), pr_dict)
        write_pickle(os.path.join(self.path, 'pk_dict.pkl'), pk_dict)

    def load_model(self, weights='best.pth'):
        """
        Load model from the disk

        :param: tag: name tag
        """
        self.model.load(weights)

    def save_model(self, tag='best'):
        """
        Save model on the disk

        :param: tag: name tag
        """
        self.model.save(
            self.model.module_name + '_' + str(tag) + '.pth',
            self.path,
            cuda_device=self.device)
