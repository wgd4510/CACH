import argparse

class BaseConfig:
    def __init__(self, args):
        super(BaseConfig, self).__init__()
        self.seed = 42  # random seed
        self.cuda_device = 'cuda:4'  # CUDA device to use

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
    parser.add_argument('--test', default=False, help='run test')
    parser.add_argument('--img-backbone', default='resnet18', type=str, 
                        help='resnet18, resnet34, resnet50, vit-base, swint-tiny, swint-base, swintv2-tiny, swintv2-base')
    parser.add_argument('--txt-backbone', default='bert', help='bert, gpt2', type=str)

    parser.add_argument('--device', default='cuda:0', help='GPU ID', type=str)
    parser.add_argument('--bit', default=64, help='hash code length', type=int)
    parser.add_argument('--batch-size', default=20, help='batch size', type=int)  # 单卡 20
    parser.add_argument('--dataset', default='rsicd', help='ucm or rsicd or sydney', type=str)
    parser.add_argument('--epochs', default=100, help='training epochs', type=int)
    parser.add_argument('--token_length', default=64, help='max token length for clean data', type=int)
    parser.add_argument('--hidden_length', default=6, help='Number of last hidden states to use', type=int)
    parser.add_argument('--hidden_operation', default='sum', help='hidden states sum or concat', type=str)
    parser.add_argument('--tag', default='CACH', help='model tag', type=str)

    parser.add_argument('--img-aug', default='aug_center', type=str, help="each_img_random、aug_center")
    parser.add_argument('--txt-aug', default='rule-based', type=str, 
                        help="rule-based、backtranslation-prob、backtranslation-chain")
    
    parser.add_argument('--optimizer', default='Adam', help='Adam optimizer', type=str)
    parser.add_argument('--feature_loss', default=True, type=bool,  
                        help='whether to calculate feature semantic similarity loss')
    parser.add_argument('--decoder_loss', default=True, type=bool,  
                        help='whether to calculate decoder reconstructed loss')
    parser.add_argument('--contrastive-weights', default=[1.0, 1.0, 1.0], nargs='+', 
                        help='contrastive loss component weights')
    parser.add_argument('--encoder_weights', default=[1.0, 1.0, 1.0, 1.0], nargs='+',
                        help='img&text、img_aug&text_aug、img&text_aug、img_aug&text weights')
    parser.add_argument('--feature_weights', default=[1.0, 1.0, 1.0], nargs='+', help='loss weights')

    parser.add_argument('--crop-size', default=[200, 200], nargs=2, type=int, 
                        help="crop size for 'center_crop' and 'random_crop'")
    parser.add_argument('--rot-deg', default=[-10, -5], nargs=2, type=int, 
                        help="random rotation degrees range for 'rotation_cc'")
    parser.add_argument('--blur-val', default=[3., 3., 1.1, 1.3], nargs=4, type=float,
                        help="gaussian blur parameters for 'blur_cc'")
    parser.add_argument('--jit-str', default=[0.5], nargs=1, type=float, 
                        help="color jitter strength for 'jitter_cc'")

    args = parser.parse_args()
    return args


args = parse_opt()
test = args.test
img_backbone = args.img_backbone
txt_backbone = args.txt_backbone
device = args.device
bit = args.bit
batch_size = args.batch_size
epochs = args.epochs
token_length = args.token_length
hidden_length = args.hidden_length
operation = args.hidden_operation
tag = args.tag
dataset = args.dataset
txt_aug = args.txt_aug
img_aug = args.img_aug
optimizer = args.optimizer
feature_loss = args.feature_loss
decoder_loss = args.decoder_loss  
contrastive_weights = args.contrastive_weights
encoder_weights = args.encoder_weights
feature_weights = args.feature_weights
acs = args.crop_size
ard = args.rot_deg
abv = args.blur_val
ajs = args.jit_str


class ConfigModel(BaseConfig):
    test = test
    backbone = img_backbone
    imgmodel_list = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'vit-base', 'swint-tiny', 'swint-base', 'swintv2-tiny', 'swintv2-base']
    imgmodel_weights = [
        '/home/wgd/code/Transformers/weights/resnet18',
        '/home/wgd/code/Transformers/weights/resnet34',
        '/home/wgd/code/Transformers/weights/resnet50',
        '/home/wgd/code/Transformers/weights/resnet101',
        '/home/wgd/code/Transformers/weights/resnet152',
        '/home/wgd/code/Transformers/weights/vit-base-patch16-224-in21k',
        '/home/wgd/code/Transformers/weights/swin-tiny-patch4-window7-224',
        '/home/wgd/code/Transformers/weights/swin-base-patch4-window7-224-in22k',
        '/home/wgd/code/Transformers/weights/swinv2-tiny-patch4-window8-256',
        '/home/wgd/code/Transformers/weights/swinv2-base-patch4-window12-192-22k'
    ]

    img_sizes = [224, 224, 224, 224, 224, 224, 224, 224, 256, 192]
    image_dims = [512, 512, 2048, 2048, 2048, 768, 768, 1024, 768, 1024]
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

    model_type = '_'.join(['CACH', img_backbone, txt_backbone])
    batch_size = batch_size
    epochs = epochs
    tag = tag
    dataset = dataset
    txt_aug = txt_aug
    img_aug = img_aug
    optimizer = optimizer
    feature_loss = feature_loss
    decoder_loss = decoder_loss
    contrastive_weights = contrastive_weights  # [inter, intra_img, intra_txt]
    encoder_weights = encoder_weights
    feature_weights = feature_weights

    ard = ard
    abv = abv
    ajs = ajs

    if dataset == 'ucm':
        label_dim = 21
        acs = [200, 200]
    elif dataset == 'rsicd':
        label_dim = 31
        acs = [200, 200]
    elif dataset == 'sydney':
        label_dim = 7
        acs = [480, 480]

    # train 0.5, val 0.1 test 0.4
    dataset_train_split = 0.5  
    dataset_query_split = 0.2  

    caption_token_length = token_length  # use hardcoded number, max token length for clean data is 40
    caption_hidden_states = hidden_length  # Number of last BERT's hidden states to use

    caption_hidden_states_operator = operation  # "How to combine hidden states: 'sum' or 'concat'"
    if caption_hidden_states_operator == 'sum':
        text_dim = 768
    elif caption_hidden_states_operator == 'concat':
        text_dim = caption_hidden_states * 768

    hidden_dim = 1024 * 4
    hash_dim = bit
    lr = 0.0001
    # lr = 0.001
    valid = True  # validation
    valid_freq = 20  # validation frequency (epochs)
    retrieval_map_k = 20
    savepath = 'runs/' + dataset + '/' + '_'.join(
        [backbone, txt_backbone, img_aug, txt_aug,
         tag, 'hidden'+str(caption_hidden_states), 'bit'+str(hash_dim)])
    img_transforms_dicts = {
        'center_crop_only': {
            'center_crop': tuple(acs)
        },
        'random_crop_only': {
            'random_crop': tuple(acs)
        },
        'aug_center': {
            'blur': ((3, 3), (1.1, 1.3)),
            'rotation': [(-10, -5)],  # [(-10, -5), (5, 10)]
            'center_crop': tuple(acs)
        },
        'rotation_cc': {
            'rotation': [tuple(ard)],
            'center_crop': tuple(acs)
        },
        'blur_cc': {
            'blur': (tuple([int(i) for i in abv[:2]]), tuple(abv[2:])),
            'center_crop': tuple(acs)
        },
        'jitter_cc': {
            'jitter': ajs[0],
            'center_crop': tuple(acs)
        },
        'each_img_random': ['rotation_cc', 'blur_cc', 'jitter_cc'],
        'no_aug': {}
    }

    def __init__(self, args):
        super(ConfigModel, self).__init__(args)
        self.cuda_device = args.device


cfg = ConfigModel(args)
