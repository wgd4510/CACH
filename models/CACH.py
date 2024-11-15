import os
import torch
from torch import nn
import transformers
from torchinfo import summary


class LBR(nn.Module):
    """(Linear => BN => ReLU) """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.lbr = nn.Sequential(
            nn.Linear(in_channels, out_channels, bias=True),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(True),
        )

    def forward(self, x):
        return self.lbr(x)
    
class Down(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.lbr1 = LBR(in_channels, in_channels // 2)
        self.lbr2 = LBR(in_channels // 2, in_channels // 4)
        self.lbr3 = LBR(in_channels // 4, in_channels // 8)

    def forward(self, x):
        x1 = self.lbr1(x)
        x2 = self.lbr2(x1)
        x3 = self.lbr3(x2)
        return x1, x2, x3
    
class Up(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.lbr1 = LBR(in_channels, in_channels * 2)
        self.lbr2 = LBR(in_channels * 2, in_channels * 4)
        self.lbr3 = LBR(in_channels * 4, in_channels * 8)

    def forward(self, x):
        x = self.lbr1(x)
        x = self.lbr2(x)
        x = self.lbr3(x)
        return x

class Gen_hash(nn.Module):
    def __init__(self, in_channels, hidden_dim, hash_dim):
        super().__init__()
        self.f_module = LBR(in_channels, hidden_dim)
        self.down = Down(hidden_dim)
        self.hash_module = nn.Sequential(nn.Linear(hidden_dim // 8, hash_dim, bias=True), nn.Tanh())

    def forward(self, x):
        x = self.f_module(x)  # [bs, image_dim] -> [bs, hidden_dim2048] 
        x1, x2, x3 = self.down(x) # [bs, hidden_dim2048] -> [bs, 1024], [bs, 512], [bs, 256]
        out = self.hash_module(x3)  # [bs, 256] -> [bs, hash_dim] 
        img_out = torch.cat((x, x1, x2, x3, out), dim=1)  # [bs, hidden_dim+1024+512+256+hash_dim]
        return img_out

class CACH_Transformer(nn.Module):
    def __init__(self, cfg):
        super(CACH_Transformer, self).__init__()
        self.module_name = 'CACH_Transformer'
        self.image_dim = cfg.image_dim
        self.text_dim = cfg.text_dim
        self.hidden_dim = cfg.hidden_dim
        self.hash_dim = cfg.hash_dim
        self.operation = cfg.caption_hidden_states_operator
        self.caption_hidden_states = cfg.caption_hidden_states

        self.hs = [i for i in range(-(cfg.caption_hidden_states), 0)]

        if cfg.backbone in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']:
            self.img_backbone = transformers.ResNetModel.from_pretrained(cfg.img_weights)
        elif cfg.backbone in ['vit-base']:
            self.img_backbone = transformers.ViTModel.from_pretrained(cfg.img_weights)
        elif cfg.backbone in ['swint-tiny', 'swint-base']:
            self.img_backbone = transformers.SwinModel.from_pretrained(cfg.img_weights)
        elif cfg.backbone in ['swintv2-tiny', 'swintv2-base']:
            self.img_backbone = transformers.Swinv2Model.from_pretrained(cfg.img_weights)
        # summary(self.img_backbone, input_size=[(8, 3, 224, 224)], device="cuda")

        self.image_module = Gen_hash(self.image_dim, self.hidden_dim, self.hash_dim)

        self.img_decoder_i2i = Up(self.hidden_dim // 8)
        self.img_decoder_i2t = Up(self.hidden_dim // 8)

        if cfg.txt_backbone in ['bert']:                                
            self.txt_encoder = transformers.BertModel.from_pretrained(cfg.txt_weights, output_hidden_states=True)
        elif cfg.txt_backbone in ['gpt2']:  
            self.txt_encoder = transformers.GPT2Model.from_pretrained(cfg.txt_weights, output_hidden_states=True)

        self.text_module = Gen_hash(self.text_dim, self.hidden_dim, self.hash_dim)

        self.text_decoder_t2t = Up(self.hidden_dim // 8)
        self.text_decoder_t2i = Up(self.hidden_dim // 8)
        # hash discriminator
        # self.hash_dis = nn.Sequential(
        #     nn.Linear(self.hash_dim, self.hash_dim * 2, bias=True),
        #     nn.BatchNorm1d(self.hash_dim * 2), nn.ReLU(True),
        #     nn.Linear(self.hash_dim * 2, 1, bias=True))

    def forward(self, *args):
        if len(args) == 4:
            res = self.forward_img_img_txt_txt(*args)
        elif len(args) == 2:
            res = self.forward_img_txt(*args)
        else:
            raise Exception('Method take wrong arguments')
        return res

    def txt_encoder2module(self, hidden_states):
        last_hidden = [hidden_states[i] for i in self.hs] 

        if self.operation == 'sum':
            # [(batch_size, max_len, 768)] -> (hidden_states, batch_size, max_len, 768)
            hiddens = torch.stack(last_hidden) 
            # (hidden_states, batch_size, max_len, 768) -> (bs, max_len, 768)
            resulting_states = torch.sum(hiddens, dim=0, keepdim=False)
        elif self.operation == 'concat': 
            # (batch_size, max_len, 768) -> (batch_size, max_len, 768 * max_len)
            resulting_states = torch.cat(tuple(last_hidden), dim=2)
        else:
            raise Exception('unknown operation ' + str(self.operation))

        # token embeddings to sentence embedding via token embeddings averaging
        # 3D (batch_size, tokens, resulting_states.shape[2]) -> 2D (batch_size, resulting_states.shape[2])
        sentence_emb = torch.mean(resulting_states, dim=1, keepdim=False)  # [bs, 768] 
        return sentence_emb

    def forward_img_img_txt_txt(self, r_img1, r_img2, r_txt1, r_txt2):  # [bs, 3, 244, 244] [bs, 2, 64]
        r_img1 = self.img_backbone(r_img1).pooler_output.squeeze()  # [bs, 3, 244, 244] -> [bs, image_dim]
        if len(r_img1.shape) == 1:
            r_img1 = r_img1.unsqueeze(0)
        img_out = self.image_module(r_img1)

        r_img2 = self.img_backbone(r_img2).pooler_output.squeeze()
        if len(r_img2.shape) == 1:
            r_img2 = r_img2.unsqueeze(0)
        img_aug_out = self.image_module(r_img2)

        r_txt1_e = self.txt_encoder(
            input_ids=r_txt1[:, 0, :], attention_mask=r_txt1[:, 1, :])  # [bs, 2, 64] -> 13 * [bs, 64, 768]
        r_txt1_e2m = self.txt_encoder2module(r_txt1_e.hidden_states)  # [bs, 64, 768] -> [bs, 768]
        txt_out = self.text_module(r_txt1_e2m)

        r_txt2_e = self.txt_encoder(
            input_ids=r_txt2[:, 0, :], attention_mask=r_txt2[:, 1, :])
        r_txt2_e2m = self.txt_encoder2module(r_txt2_e.hidden_states)
        txt_aug_out = self.text_module(r_txt2_e2m)

        return img_out, img_aug_out, txt_out, txt_aug_out

    def forward_img_txt(self, r_img, r_txt):
        r_img = self.img_backbone(r_img).pooler_output.squeeze()
        if len(r_img.shape) == 1:
            r_img = r_img.unsqueeze(0)
        img_out = self.image_module(r_img)

        r_txt_e = self.txt_encoder(
            input_ids=r_txt[:, 0, :], attention_mask=r_txt[:, 1, :])  # [bs, 2, 64] -> 13 * [bs, 64, 768]
        r_txt_e2m = self.txt_encoder2module(r_txt_e.hidden_states)  # [bs, 64, 768] -> [bs, 768]
        txt_out = self.text_module(r_txt_e2m)
        return img_out, txt_out

    def img_refactoring(self, x):
        return self.img_decoder(x)

    def txt_refactoring(self, x):
        return self.text_decoder(x)
    
    def generate_img_code(self, x):   
        x = self.img_backbone(x).pooler_output.squeeze()  # [bs,3,244,244] -> [bs,512,1,1] -> [bs, image_dim]
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        img_out = self.image_module(x)
        return img_out.detach()
    
    def generate_txt_code(self, x):
        r_txt_e = self.txt_encoder(input_ids=x[:, 0, :], attention_mask=x[:, 1, :])
        r_txt_e2m = self.txt_encoder2module(r_txt_e.hidden_states)
        txt_out = self.text_module(r_txt_e2m)
        return txt_out.detach()

    def load(self, path, use_gpu=False):
        if not use_gpu:
            self.load_state_dict(
                torch.load(path, map_location=lambda storage, loc: storage))
        else:
            self.load_state_dict(torch.load(path))

    def save(self, name=None, path='./checkpoints', cuda_device=None):
        if not os.path.exists(path):
            os.makedirs(path)
        if cuda_device.type == 'cpu':
            torch.save(self.state_dict(), os.path.join(path, name))
        else:
            with torch.cuda.device(cuda_device):
                torch.save(self.state_dict(), os.path.join(path, name))
        return name
