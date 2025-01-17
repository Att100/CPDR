import paddle
import paddle.nn.functional as F
import paddle.optimizer as optim
import os
import random
import numpy as np
import argparse

from manager import Model, Config, Criterion, \
    Logger, LrScheduler, build_dataloaders
from utils.pbar import PBar


def update_batch_pbar(bar: PBar, steps, e, i, loss, mae):
    if i != steps-1:
        bar.updateBar(
                i+1, headData={'Epoch':e, 'Status':'training'}, 
                endData={
                    'Train loss': "{:.5f}".format(loss),
                    'Train MAE': "{:.5f}".format(mae)})
    else:
        bar.updateBar(
                i+1, headData={'Epoch':e, 'Status':'finished'}, 
                endData={
                    'Train loss': "{:.5f}".format(loss),
                    'Train MAE': "{:.5f}".format(mae)})
        
def update_test_pbar(bar: PBar, steps, e, i, mae):
    if i !=steps-1:
        bar.updateBar(
                i+1, headData={'Epoch (Test)':e, 'Status':'testing'}, 
                endData={'Test MAE': "{:.5f}".format(mae)})
    else: 
        bar.updateBar(
                i+1, headData={'Epoch (Test)':e, 'Status':'finished'}, 
                endData={'Test MAE': "{:.5f}".format(mae)})

def mae(pred, label):
    return float(paddle.mean(paddle.abs(label-pred)))

def main(args):
    # Load Config
    if args.cmd:
        configs = Config.load_all_configs()
        names = list(configs.keys())
        print("Please input the id of config you want to train")
        for i, config_name in enumerate(names):
            print("{}. {}".format(i+1, config_name))
        selected = int(input("> "))-1
        config = configs[names[selected]]
        print("---\nTraining with config: {}\n".format(names[selected]))
        Config.print_config(config, skip=['block_cfgs'])
        print("---")
    else:
        config = Config.load_config(args.cfg)

    # Set seed
    paddle.seed(999)
    random.seed(999)
    np.random.seed(999)
    os.environ['FLAGS_cudnn_deterministic'] = "True"  # CUDNN FLAG (use fix conv)

    # Prepare for Training
    model = Model.make(
        config.get('structure').get('select'),
        config.get('structure'), 
        config.get('training').get('pretrained'),
        config.get('training').get('backbone_pretrained_path'))
    criterion = Criterion.make(
        config.get('training').get('criterion').get('select'),
        **config.get('training').get('criterion').get('params'))
    train_loader, test_loader = build_dataloaders(config)
    train_steps, test_steps = len(train_loader), len(test_loader)
    total_epochs = config.get('training').get('epochs')
    lr_sch = LrScheduler.make(
        config.get('training').get('lr').get('select'),
        **config.get('training').get('lr').get('params')
    )
    optimizer = optim.Adam(
        lr_sch.scheduler,
        parameters=model.parameters(),
        weight_decay=config.get('training').get('weight_decay')
    )
    logger = Logger.from_config(config)
    

    # Start Training
    for e in range(total_epochs):
        train_loss = 0
        train_mae = 0
        test_mae = 0

        bar = PBar(maxStep=train_steps)
        model.train()
        
        # train iters
        for i, (image, label) in enumerate(train_loader()):
            optimizer.clear_grad()
            pred = model(image)

            loss = criterion(pred, label)
            pred = F.sigmoid(paddle.squeeze(pred[0], 1))

            loss.backward()
            optimizer.step()

            batch_loss = loss.numpy()[0]
            batch_mae = mae(pred, label)
            train_loss += batch_loss
            train_mae += batch_mae

            update_batch_pbar(bar, train_steps, e+1, i, train_loss/(i+1), train_mae/(i+1))
            logger.update_train(e+1, e*train_steps+i+1, batch_loss, batch_mae)
            
            lr_sch.step_iter()
        lr_sch.step_epoch()

        if (e+1) % config.get('training').get('test_interval') == 0:
            bar = PBar(maxStep=test_steps)
            model.eval()

            # test iters
            for i, (image, label, h, w) in enumerate(test_loader()):
                pred = model(image)
                pred = F.interpolate(pred[0], (h, w), mode='bilinear', align_corners=True)
                
                test_mae += mae(F.sigmoid(paddle.squeeze(pred, 1)), label)

                update_test_pbar(bar, train_steps, e+1, i, test_mae/(i+1))
            
            logger.update_test(e+1, (e+1)*train_steps, test_mae/test_steps)


        if (e+1) % config.get('training').get('test_interval') == 0:
            paddle.save(
                model.state_dict(), 
                os.path.join(
                    config.get('training').get('checkpoints_path'),
                    '{}_e{}.pdparams'.format(config.get('name'), e+1)
                )
            )

    paddle.save(
        model.state_dict(), 
        os.path.join(
            config.get('training').get('weights_path'),
            '{}.pdparams'.format(config.get('name'))
        )
    )

            

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-cmd', action="store_true", default=False, 
        help="select config to run from shell")
    parser.add_argument(
        '--cfg', type=str, default='fpn_cpdr_b_mobilenetv2lite_dice_iou', 
        help="the config to train (default: fpn_cpdr_b_mobilenetv2lite_dice_iou)")

    args = parser.parse_args()

    main(args)

