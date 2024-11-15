# 测试不同隐藏层数目对结果影响（经测试，最佳为hidden_length 6）
# python main.py --dataset rsicd --bit 64 --img-backbone resnet18 --txt-backbone bert --hidden_length 1 --hidden_operation 'concat' --img-aug each_img_random --tag concat
# python main.py --dataset rsicd --bit 64 --img-backbone resnet18 --txt-backbone bert --hidden_length 2 --hidden_operation 'concat' --img-aug each_img_random --tag concat
# python main.py --dataset rsicd --bit 64 --img-backbone resnet18 --txt-backbone bert --hidden_length 3 --hidden_operation 'concat' --img-aug each_img_random --tag concat
# python main.py --dataset rsicd --bit 64 --img-backbone resnet18 --txt-backbone bert --hidden_length 4 --hidden_operation 'concat' --img-aug each_img_random --tag concat
# python main.py --dataset rsicd --bit 64 --img-backbone resnet18 --txt-backbone bert --hidden_length 6 --hidden_operation 'concat' --img-aug each_img_random --tag concat
# python main.py --dataset rsicd --bit 64 --img-backbone resnet18 --txt-backbone bert --hidden_length 8 --hidden_operation 'concat' --img-aug each_img_random --tag concat

# 测试不同图像增强方式（经测试，最佳为aug_center）
# python main.py --dataset rsicd --bit 64 --img-backbone resnet18 --txt-backbone bert --img-aug each_img_random
# python main.py --dataset rsicd --bit 64 --img-backbone resnet18 --txt-backbone bert --img-aug center_crop_only
# python main.py --dataset rsicd --bit 64 --img-backbone resnet18 --txt-backbone bert --img-aug random_crop_only
# python main.py --dataset rsicd --bit 64 --img-backbone resnet18 --txt-backbone bert --img-aug aug_center

# 测试不同文本增强方式(backtranslation最佳，考虑到 backtranslation 的计算时间复杂度高耗资源，所以默认rule-based)
# python main.py --dataset rsicd --bit 64 --img-backbone resnet18 --txt-backbone bert --txt-aug rule-based
# python main.py --dataset rsicd --bit 64 --img-backbone resnet18 --txt-backbone bert --txt-aug backtranslation-prob
# python main.py --dataset rsicd --bit 64 --img-backbone resnet18 --txt-backbone bert --txt-aug backtranslation-chain


# 测试不同损失函数影响
# CACH-CL：只有图像-文本模态内对比损失
# python main.py --dataset rsicd --bit 64 --img-backbone resnet18 --txt-backbone bert --feature_loss False --decoder_loss False --contrastive-weights 1.0 0.0 0.0 --tag CACH-CL

# CACH-CIL：模态内对比损失 + 模态间图像损失
# python main.py --dataset rsicd --bit 64 --img-backbone resnet18 --txt-backbone bert --feature_loss False --decoder_loss False --contrastive-weights 1.0 1.0 0.0 --tag CACH-CIL

# CACH-CTL: 模态内对比损失 + 模态间文本损失
# python main.py --dataset rsicd --bit 64 --img-backbone resnet18 --txt-backbone bert --feature_loss False --decoder_loss False --contrastive-weights 1.0 0.0 1.0 --tag CACH-CTL

# CACH-CITL: 模态内对比损失 + 模态间图像和文本损失
# python main.py --dataset rsicd --bit 64 --img-backbone resnet18 --txt-backbone bert --feature_loss False --decoder_loss False --contrastive-weights 1.0 1.0 1.0 --tag CACH-CITL

# CACH-CITL-SL：对比损失+语义相似损失
python main.py --dataset rsicd --bit 64 --img-backbone resnet18 --txt-backbone bert --decoder_loss False --tag CACH-CITL-SL-0.001featureloss


# CACH-CITL-RL：对比损失+重建损失
# python main.py --dataset rsicd --bit 64 --img-backbone resnet18 --txt-backbone bert --feature_loss False --tag CACH-CITL-RL


# 测试不同Bit值的影响
# python main.py --dataset rsicd --bit 16 --img-backbone resnet18 --txt-backbone bert --hidden_length 4 
# python main.py --dataset rsicd --bit 32 --img-backbone resnet18 --txt-backbone bert --hidden_length 4 
# python main.py --dataset rsicd --bit 64 --img-backbone resnet18 --txt-backbone bert --hidden_length 4 
# python main.py --dataset rsicd --bit 128 --img-backbone resnet18 --txt-backbone bert --hidden_length 4 


# 不同backbone的影响 resnet18, resnet34, resnet50, vit-base, swint-tiny, swint-base, swintv2-tiny, swintv2-base
# python main.py --dataset ucm --bit 32 --img-backbone vit-base --txt-backbone gpt2 --device cuda:1
# python main.py --dataset ucm --bit 32 --img-backbone swint-base --txt-backbone gpt2 --device cuda:1
# python main.py --dataset ucm --bit 32 --img-backbone swintv2-base --txt-backbone gpt2 --device cuda:1
# python main.py --dataset ucm --bit 32 --img-backbone resnet18 --txt-backbone gpt2 --device cuda:1
# python main.py --dataset ucm --bit 32 --img-backbone resnet50 --txt-backbone gpt2 --device cuda:1

# python main.py --dataset ucm --bit 32 --img-backbone resnet50 --txt-backbone bert --device cuda:1
# python main.py --dataset ucm --bit 32 --img-backbone vit-base --txt-backbone bert --device cuda:1
# python main.py --dataset ucm --bit 32 --img-backbone swint-base --txt-backbone bert --device cuda:1
# python main.py --dataset ucm --bit 32 --img-backbone swintv2-base --txt-backbone bert --device cuda:1

