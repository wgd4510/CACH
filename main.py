from utils.log import load_logger
from datasets.data_handlers_img_txt import DataHandler_Train, get_dataloaders
from datasets.datasets_img_txt import Dataset_img_img_txt_txt, Dataset_img_txt
from controller import Controller
from configs.config import cfg


def main(cfg):
    log = load_logger(cfg)
    mode = 'TEST' if cfg.test else 'TRAIN'
    s = 'Init ({}): Model: {}, Datasets: {}, Bits: {} , tag: {}, img_aug: {}, txt_aug: {}'
    log.info(
        s.format(mode, cfg.model_type, cfg.dataset.upper(), cfg.hash_dim, cfg.tag, cfg.img_aug, cfg.txt_aug))
    
    data_handler = DataHandler_Train
    ds_train = Dataset_img_img_txt_txt
    ds_query = Dataset_img_txt
    ds_db = Dataset_img_txt
    controller = Controller

    dl_train, dl_q, dl_db = get_dataloaders(data_handler, ds_train, ds_query, ds_db)

    controller = controller(log, cfg, (dl_train, dl_q, dl_db))

    if cfg.test:
        weights = '/home/wgd/code/Cross-modal-retrieval/CACH/runs/rsicd/resnet18_bert_aug_center_backtranslation-prob_CACH_hidden4_bit64/CACH_Transformer_best.pth'
        controller.load_model(weights)
        controller.test()
    else:
        for epoch in range(cfg.epochs):
            controller.train_epoch(epoch)
            if ((epoch + 1) % cfg.valid_freq == 0) and cfg.valid:
                controller.eval(epoch)
            # save the model
            if epoch + 1 == cfg.epochs:
                controller.save_model('last')
                controller.training_complete()

    s = 'Done ({}): Model: {}, Datasets: {}, Bits: {} , tag: {}, img_aug: {}, txt_aug: {}'            
    log.info(
        s.format(mode, cfg.model_type, cfg.dataset.upper(), cfg.hash_dim, cfg.tag, cfg.img_aug, cfg.txt_aug))           


if __name__ == '__main__':
    main(cfg)
