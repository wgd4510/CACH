from torch.utils.data import DataLoader
from utils.log import load_logger
from datasets.data_handlers_img_txt import DataHandler_Infer
from controller_infer import Controller, Dataset_img_txt
from configs.config_infer import cfg
 

def infer(cfg):
    log = load_logger(cfg)

    s = 'Init : Datasets: {}, Bits: {} , tag: {}'
    log.info(s.format(cfg.dataset.upper(), cfg.hash_dim, cfg.tag))
    
    data_handler = DataHandler_Infer()
    db_tuple = data_handler.load_database()

    dataset_db = Dataset_img_txt(*db_tuple)
    dataloader_db = DataLoader(dataset_db, batch_size=cfg.batch_size)

    controller = Controller(log, cfg, dataloader_db, data_handler)
    weights = '/home/wgd/code/Cross-modal-retrieval/CACH/runs/rsicd/resnet18_bert_aug_center_backtranslation-chain_CACH_hidden4_bit64/CACH_Transformer_best.pth'
    log.info(f'Load weights: {weights}')
    controller.load_model(weights)
    controller.infer_once()
    controller.infer()


if __name__ == '__main__':
    infer(cfg)
