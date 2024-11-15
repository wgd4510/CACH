import transformers
from torchinfo import summary

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
backbone = 'resnet152'

img_weights = imgmodel_weights[imgmodel_list.index(backbone)]
imgs = img_sizes[imgmodel_list.index(backbone)]

if backbone in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']:
    img_backbone = transformers.ResNetModel.from_pretrained(img_weights)
elif backbone in ['vit-base']:
    img_backbone = transformers.ViTModel.from_pretrained(img_weights)
elif backbone in ['swint-tiny', 'swint-base']:
    img_backbone = transformers.SwinModel.from_pretrained(img_weights)
elif backbone in ['swintv2-tiny', 'swintv2-base']:
    img_backbone = transformers.Swinv2Model.from_pretrained(img_weights)

summary(img_backbone, input_size=[(1, 3, imgs, imgs)], device="cuda")

