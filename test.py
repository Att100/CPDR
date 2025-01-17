import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from paddle.io import DataLoader
import argparse
import os

from manager import Config, Model
from utils.dataset import DATASET_TEST
from utils.metrics import FullMetrics


def main(args):
    # Load Config
    if args.cmd:
        configs = Config.load_all_configs()
        names = list(configs.keys())
        print("Please input the id of config you want to test")
        for i, config_name in enumerate(names):
            print("{}. {}".format(i+1, config_name))
        selected = int(input("> "))-1
        config = configs[names[selected]]
        print("---\nTesting with config: {}\n".format(names[selected]))
        Config.print_config(config, skip=['block_cfgs'])
        print("---")
    else:
        config = Config.load_config(args.cfg)
    Config.print_config(config, skip=['block_cfgs'])

    # Make Model & Load Checkpoint
    model = Model.make(
        config.get('structure').get('select'),
        config.get('structure'), False)
    model.set_state_dict(paddle.load(
        os.path.join(
            config.get('training').get('weights_path'),
            f"{config['name']}.pdparams")))
    model.eval()
    
    # Evaluation
    test_set = DATASET_TEST(
        args.dataset_root, 
        args.dataset_name)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
    metrics = FullMetrics()

    for img, gt, h, w in tqdm(test_loader()):
        pred = model(img)
        pred = F.interpolate(pred[0], (h, w), mode='bilinear', align_corners=True)
        pred = (F.sigmoid(paddle.squeeze(pred, 1))[0].numpy() * 255).astype(np.float32)
        metrics.eval_step(pred, gt[0].numpy()*255)
    for k, v in metrics.result_all().items():
        print(f"{k}: {v}")
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-cmd', action="store_true", default=False, 
        help="select config to run from shell")
    parser.add_argument(
        '--cfg', type=str, default='fpn_cpdr_b_mobilenetv2lite_dice_iou', 
        help="the config to train (default: fpn_cpdr_b_mobilenetv2lite_dice_iou)")
    parser.add_argument(
        '--ckpt_path', type=str, default="-",
        help="path of weights (used to overide weights dir defined in config)"
    )
    parser.add_argument(
        '--dataset_root', type=str, default="./dataset/DUTS/DUTS-TE",
        help="root dir of the dataset used for testing"
    )
    parser.add_argument(
        '--dataset_name', type=str, default="DUTS-TE",
        help="name of the dataset used for testing"
    )
    
    args = parser.parse_args()
    main(args)
