import argparse

class BaseConfig:
    def __init__(self, args):
        super(BaseConfig, self).__init__()
        self.seed = 42  # random seed
        self.cuda_device = 'cuda:0'  # CUDA device to use

        self.dataset = args.dataset.lower()
        self.dataset_json_file = "./data/augmented_{}.json".format(args.dataset.upper())

        if args.dataset == 'ucm':
            self.dataset_image_folder_path = "/home/wgd/code/Cross-modal-retrieval/Datasets/UCM_Captions/images"
        if args.dataset == 'rsicd':
            self.dataset_image_folder_path = "/home/wgd/code/Cross-modal-retrieval/Datasets/RSICD/images"
        if args.dataset == 'sydney':
            self.dataset_image_folder_path = "/home/wgd/code/Cross-modal-retrieval/Datasets/Sydney_captions/images"
        # self._print_config()

    def _print_config(self):
        print('Configuration:', self.__class__.__name__)
        for v in self.__dir__():
            if not v.startswith('_'):
                print('\t{0}: {1}'.format(v, getattr(self, v)))


def parse_opt():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--img-backbone', default='resnet18', type=str, 
                        help='resnet18, resnet34, resnet50, vit-base, swint-tiny, swint-base, swintv2-tiny, swintv2-base')
    parser.add_argument('--txt-backbone', default='bert', help='bert, gpt2', type=str)

    parser.add_argument('--device', default='cuda:0', help='GPU ID', type=str)
    parser.add_argument('--bit', default=64, help='hash code length', type=int)
    parser.add_argument('--batch-size', default=20, help='batch size', type=int)  # 单卡 20
    parser.add_argument('--dataset', default='rsicd', help='ucm or rsicd or sydney', type=str)
    parser.add_argument('--token_length', default=64, help='max token length for clean data', type=int)
    parser.add_argument('--hidden_length', default=6, help='Number of last hidden states to use', type=int)
    parser.add_argument('--tag', default='CACH', help='model tag', type=str)
    
    args = parser.parse_args()
    return args


args = parse_opt()
img_backbone = args.img_backbone
txt_backbone = args.txt_backbone
device = args.device
bit = args.bit
batch_size = args.batch_size
token_length = args.token_length
hidden_length = args.hidden_length
tag = args.tag
dataset = args.dataset


class ConfigModel(BaseConfig):
    backbone = img_backbone
    imgmodel_list = ['resnet18', 'resnet34', 'resnet50', 'vit-base', 'swint-tiny', 'swint-base', 'swintv2-tiny', 'swintv2-base']
    imgmodel_weights = [
        '/home/wgd/code/Transformers/weights/resnet18',
        '/home/wgd/code/Transformers/weights/resnet34',
        '/home/wgd/code/Transformers/weights/resnet50',
        '/home/wgd/code/Transformers/weights/vit-base-patch16-224-in21k',
        '/home/wgd/code/Transformers/weights/swin-tiny-patch4-window7-224',
        '/home/wgd/code/Transformers/weights/swin-base-patch4-window7-224-in22k',
        '/home/wgd/code/Transformers/weights/swinv2-tiny-patch4-window8-256',
        '/home/wgd/code/Transformers/weights/swinv2-base-patch4-window12-192-22k'
    ]

    img_sizes = [224, 224, 224, 224, 224, 224, 256, 192]
    image_dims = [512, 512, 2048, 768, 768, 1024, 768, 1024]
    img_weights = imgmodel_weights[imgmodel_list.index(backbone)]
    imgsize = img_sizes[imgmodel_list.index(backbone)]
    image_dim = image_dims[imgmodel_list.index(backbone)]

    txt_backbone = txt_backbone
    txtmodel_list = ['bert', 'gpt2']
    txtmodel_weights = [
        '/home/wgd/code/Transformers/weights/bert-base-uncased',
        '/home/wgd/code/Transformers/weights/gpt2',
    ]
    txt_weights = txtmodel_weights[txtmodel_list.index(txt_backbone)]

    # model_type = '_'.join(['CACH', img_backbone, txt_backbone])
    batch_size = batch_size
    tag = tag
    dataset = dataset
    caption_token_length = token_length  # use hardcoded number, max token length for clean data is 40
    caption_hidden_states = hidden_length  # Number of last BERT's hidden states to use
    caption_hidden_states_operator = 'sum'  # "How to combine hidden states: 'sum' or 'concat'"
    if caption_hidden_states_operator == 'sum':
        text_dim = 768
    elif caption_hidden_states_operator == 'concat':
        text_dim = caption_hidden_states * 768

    hidden_dim = 1024 * 4
    hash_dim = bit

    savepath = 'runs_test/' + dataset + '/' + '_'.join(
        [backbone, txt_backbone, tag, 'hidden'+str(caption_hidden_states), 'bit'+str(hash_dim)])

    def __init__(self, args):
        super(ConfigModel, self).__init__(args)
        self.cuda_device = args.device


cfg = ConfigModel(args)
