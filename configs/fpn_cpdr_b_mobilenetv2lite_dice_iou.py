config = {
    'name': "fpn_cpdr_b_mobilenetv2lite_dice_iou",
    'structure': {
        'select': 'fpn_with_neck',
        'backbone': {
            'select': "mobilenet_v2lite"
        },
        'mappers': {
            'block_cfgs': [
                ['mapr1', 'conv2d_bn_relu', [160, 128, 1]],
                ['mapr2', 'conv2d_bn_relu', [64, 128, 1]],
                ['mapr3', 'conv2d_bn_relu', [32, 64, 1]],
                ['mapr4', 'conv2d_bn_relu', [24, 32, 1]],
                ['mapr5', 'conv2d_bn_relu', [32, 16, 1]]
            ]
        },
        'decoders': {
            'block_cfgs': [
                ['dec1', 'decoder_c', [128, 64, 64]],
                ['dec2', 'decoder_c', [64, 32, 64]],
                ['dec3', 'decoder_c', [32, 16, 32]],
                ['dec4', 'decoder_c', [16, 16, 16]],
            ]
        },
        'neck': {
            'select': 'cpdr_b',
            'block_cfgs': [
                ['dacf1', 'dacf', [64, 8, 7]],
                ['dacf2', 'dacf', [32, 4, 7]],
                ['dacf3', 'dacf', [16, 2, 7]],
            ],
        },
        'heads': {
            'block_cfgs': [
                ['classifier1', 'conv2d', [8, 1, 3, 1, 1]],
                ['classifier2', 'conv2d', [16, 1, 3, 1, 1]],
                ['classifier3', 'conv2d', [32, 1, 3, 1, 1]]
            ]
        }
    },
    'training': {
        'pretrained': True,
        'backbone_pretrained_path': None,
        'pretrained_from_paddle': True,
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
                'total_steps': 40 * 660,
                'gamma': 3
            }
        },
        'epochs': 40,
        'test_interval': 5,
        'num_workers': 8,
        'weight_decay': 5e-4
    },
    'testing': {
        
    }
}