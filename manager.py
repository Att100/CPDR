import paddle
import importlib
import os
import copy
from paddle.io import DataLoader

from models.backbones import Backbone
from models.blocks import Block
from models.decoders import Decoder
from models.necks import Neck
from models.model import Model
from models.criterions import Criterion

from utils.dataset import PreLoadedDUTS_TR, DUTS_TR, DATASET_TEST


class Config:
    @staticmethod
    def load_all_configs():
        configs = dict()
        for dir in os.listdir("./configs"):
            if dir.split(".")[-1] == 'py':
                name = dir.split(".")[0]
                configs[name] = Config.load_config(name)
        return configs
    
    @staticmethod
    def load_config(name):
        return importlib.import_module("configs.{}".format(name)).config
    
    @staticmethod
    def print_config(config: dict, skip=[], nspace=2, skip_replace="..."):
        def _print_config(_config, level):
            if isinstance(_config[1], dict):
                if level > 0 and _config[0] is not None:
                    indent = " " * nspace * (level-1)
                    print(indent + "- {}".format(_config[0]))
                for item in _config[1].items():
                    _print_config(item, level+1)
            else:
                if level > 1:
                    indent = " " * nspace * (level-1)
                    print(indent + "- {}: {}".format(
                        _config[0], 
                        _config[1] if _config[0] not in skip else skip_replace))
                else:
                    print("{}: {}".format(
                        _config[0], 
                         _config[1] if _config[0] not in skip else skip_replace))
        _print_config([None, config], 0)

    @staticmethod
    def apply_params(config: dict, new_configs: dict):
        config = copy.deepcopy(config)
        def _recursive(_config, _new_config: dict):
            for k, v in _new_config.items():
                if k in _config.keys() and isinstance(v, dict):
                    _recursive(_config[k], v)
                else:
                    _config[k] = v
        _recursive(config, new_configs)
        return config


class Logger:
    def __init__(self, path, name) -> None:
        if not os.path.exists(path):
            os.makedirs(path)
        
        self.train_log = open(os.path.join(path, "{}_train.txt".format(name)), 'w')
        self.test_log = open(os.path.join(path, "{}_test.txt".format(name)), 'w')

        self.train_log.write("epoch,iter,loss,mae\n")
        self.test_log.write("epoch,iter,mae\n")
    
    @staticmethod
    def from_config(config):
        return Logger(
            config.get('training').get('log_path'),
            config.get('name'))

    def close(self):
        self.train_log.close()
        self.test_log.close()

    def update_train(self, e, i, loss, mae):
        self.train_log.write("{},{},{},{}\n".format(e, i, loss, mae))
    
    def update_test(self, e, i, mae):
        self.test_log.write("{},{},{}\n".format(e, i, mae))


class LrScheduler:
    modules = dict()

    @staticmethod
    def register(classobj):
        LrScheduler.modules[classobj.name] = classobj

    @staticmethod
    def register_with_name(classobj, name: str):
        LrScheduler.modules[name] = classobj
    
    @staticmethod
    def get(name):
        return LrScheduler.modules.get(name)
    
    @staticmethod
    def make(name, *args, **kwargs):
        return LrScheduler.get(name)(*args, **kwargs)


@LrScheduler.register
class MultiStepDecay:
    name = "multistep_decay"

    def __init__(self, lr, milestones, gamma=0.1, verbose=True) -> None:
        self.scheduler = paddle.optimizer.lr.MultiStepDecay(lr, milestones, gamma, verbose=verbose)

    def step_epoch(self):
        self.scheduler.step()

    def step_iter(self):
        pass


@LrScheduler.register
class ExponentialDecay:
    name = "exponential_decay"

    def __init__(self, lr, gamma=0.9, verbose=True) -> None:
        self.scheduler = paddle.optimizer.lr.ExponentialDecay(learning_rate=lr, gamma=gamma, verbose=verbose)

    def step_epoch(self):
        self.scheduler.step()

    def step_iter(self):
        pass


@LrScheduler.register
class LinearWarmupPoly:
    name = 'linear_warmup_poly'

    def __init__(self, lr, warmup_steps, total_steps, gamma=0.9) -> None:
        def poly_with_warmup(step):
            # poly-lr with linear warmup
            recenter = step - warmup_steps
            decay_ratio = 1 - ((step-warmup_steps) / (total_steps-warmup_steps))**gamma
            return min(step, warmup_steps) / warmup_steps * (1 if recenter <=0 else decay_ratio)
        self.scheduler = paddle.optimizer.lr.LambdaDecay(lr, poly_with_warmup)

    def step_epoch(self):
        pass

    def step_iter(self):
        self.scheduler.step()


@LrScheduler.register
class LinearWarmUpExponential:
    name = 'linear_warmup_exponential'

    def __init__(self, lr, warmup_steps, gamma=0.99984) -> None:
        def exponential_with_warmup(step):
            # exponential-lr with linear warmup
            recenter = step - warmup_steps
            decay_ratio = gamma ** recenter
            return min(step, warmup_steps) / warmup_steps * (1 if recenter <=0 else decay_ratio)
        self.scheduler = paddle.optimizer.lr.LambdaDecay(lr, exponential_with_warmup)

    def step_epoch(self):
        pass

    def step_iter(self):
        self.scheduler.step()
    
    
def build_dataloaders(config: dict):
    if not config.get('training').get('preload_dataset', False):
        train_set = DUTS_TR(config.get('training').get('dataset_path'))
    else: 
        train_set = PreLoadedDUTS_TR(config.get('training').get('dataset_path'))
        
    test_set = DATASET_TEST(
        os.path.join(
            config.get('training').get('dataset_path'), 
            config.get('training').get('test_set')), 
        config.get('training').get('test_set'))
    train_loader = DataLoader(
        train_set, 
        batch_size=config.get('training').get('batch_size'), 
        shuffle=True, 
        num_workers=config.get('training').get('num_workers'))
    test_loader = DataLoader(
        test_set, 
        batch_size=1, 
        shuffle=False)
    return train_loader, test_loader


