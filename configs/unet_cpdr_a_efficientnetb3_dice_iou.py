config = {
    'name': "unet_cpdr_a_efficientnetb3_dice_iou",
    'structure': {
        'select': 'fpn_with_neck',
        'backbone': {
            'select': "efficientnet_b3"
        },
        'mappers': {
            'block_cfgs': [
                ['mapr1', 'conv2d_bn_relu', [1536, 128, 1]],
                ['mapr2', 'conv2d_bn_relu', [136, 128, 1]],
                ['mapr3', 'conv2d_bn_relu', [48, 64, 1]],
                ['mapr4', 'conv2d_bn_relu', [32, 32, 1]],
                ['mapr5', 'conv2d_bn_relu', [40, 16, 1]]
            ]
        },
        'decoders': {
            'block_cfgs': [
                ['dec1', 'decoder_b', [128, 128, 1, 2]],
                ['dec2', 'decoder_b', [128, 64, 2, 2]],
                ['dec3', 'decoder_b', [64, 32, 2, 2]],
                ['dec4', 'decoder_b', [32, 16, 2, 1]],
            ]
        },
        'neck': {
            'select': 'cpdr_a',
            'block_cfgs': [
                ['conv_x8_down', 'conv2d_bn_relu', [16, 16, 3, 1, 2]],
                ['conv_x4_down', 'conv2d_bn_relu', [32, 32, 3, 1, 2]],
                ['conv', 'conv2d_bn_relu', [16, 16, 3, 1]],
                ['conv_x8', 'conv2d_bn_relu', [32, 32, 3, 1]],
                ['conv_x4', 'conv2d_bn_relu', [64, 64, 3, 1]],
                ['conv_x4_up', 'conv2d_bn_relu', [64, 32, 3, 1]],
                ['conv_x8_up', 'conv2d_bn_relu', [32, 16, 3, 1]],
                ['conv2', 'conv2d_bn_relu', [32, 16, 3, 1]],
                ['conv2_x8', 'conv2d_bn_relu', [64, 32, 3, 1]],
                ['conv2_x4', 'conv2d_bn_relu', [64, 64, 3, 1]],
                ['attn_x8', 'se', [16, 2, 2]],
                ['attn_x4', 'se', [32, 4, 2]],
                ['attn_x2', 'se', [64, 8, 2]],
                ['attn2_x4', 'cbam-sp', [7]],
                ['attn2_x8', 'cbam-sp', [7]]
            ],
        },
        'heads': {
            'block_cfgs': [
                ['classifier1', 'conv2d', [16, 1, 3, 1, 1]],
                ['classifier2', 'conv2d', [32, 1, 3, 1, 1]],
                ['classifier3', 'conv2d', [64, 1, 3, 1, 1]]
            ]
        }
    },
    'training': {
        'pretrained': True,
        'backbone_pretrained_path': "./ckpts/pretrained/efficientnet-b3-5fb5a3c3.pdparams",
        'pretrained_from_paddle': False,
        'criterion': {
            'select': 'ms_dice_iou',
            'params': {'w1': [1, 0.8, 0.5], 'w2': [1, 1, 1]}
        },
        'preload_dataset': True,
        'dataset_path': "./dataset/DUTS",
        'checkpoints_path': "./ckpts/_ckpts",
        'weights_path': "./ckpts",
        'log_path': "./logs",
        'test_set': 'DUTS-TE',
        'batch_size': 16,
        'lr': {
            'select': 'linear_warmup_poly',
            'params': {
                'lr': 5e-5, 
                'warmup_steps': 5 * 660,
                'total_steps': 20 * 660,
                'gamma': 5
            }
        },
        'epochs': 20,
        'test_interval': 5,
        'num_workers': 8,
        'weight_decay': 5e-4
    },
    'testing': {
        
    }
}