# Created by Haiping Lu from modifying https://github.com/thuml/CDAN/blob/master/pytorch/train_image.py
import os
import argparse
import warnings
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import kale.utils.da_logger as da_logger  # np error if move this to later, not sure why
import torch
import pytorch_lightning as pl

from config import get_cfg_defaults
from model import get_model
from kale.loaddata.digits_access import DigitDataset
from kale.loaddata.multisource import MultiDomainDatasets


# import kale.utils.seed as seed # to unify later used pl seed_everything

def arg_parse():
    parser = argparse.ArgumentParser(description='Domain Adversarial Networks on Digits Datasets')
    parser.add_argument('--cfg', required=True, help='path to config file', type=str)
    # parser.add_argument('--output', default='default', help='folder to save output', type=str)
    parser.add_argument('--gpus', default='0', help='gpu id(s) to use', type=str)
    parser.add_argument('--resume', default='', type=str)
    args = parser.parse_args()
    return args


def main():
    args = arg_parse()
    # ---- setup device ---- NO NEED for Lightning, handled by Lightning
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # print('==> Using device ' + device)

    # ---- setup configs ----
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg)
    cfg.freeze()
    # torch.manual_seed(cfg.SOLVER.SEED)
    # seed.set_seed(cfg.SOLVER.SEED)

    # ---- setup output ----    
    os.makedirs(cfg.OUTPUT.DIR, exist_ok=True)

    # ---- setup dataset ----
    source, target, num_channels = DigitDataset.get_source_target(DigitDataset(cfg.DATASET.SOURCE.upper()),
                                                                  DigitDataset(cfg.DATASET.TARGET.upper()),
                                                                  cfg.DATASET.ROOT)
    dataset = MultiDomainDatasets(source, target, config_weight_type=cfg.DATASET.WEIGHT_TYPE,
                                  config_size_type=cfg.DATASET.SIZE_TYPE)

    # ---- setup model and logger ----
    print('==> Building model..')
    model, train_params = get_model(cfg, dataset, num_channels)
    logger, results, checkpoint_callback, test_csv_file = da_logger.setup_logger(train_params, cfg.OUTPUT.DIR,
                                                                                 cfg.DAN.METHOD)

    # Repeat multiple times to get std
    for i in range(0, cfg.DATASET.NUM_REPEAT):
        seed = cfg.SOLVER.SEED + i
        pl.seed_everything(seed)
        trainer = pl.Trainer(
            progress_bar_refresh_rate=cfg.OUTPUT.PB_FRESH,  # in steps
            min_epochs=cfg.SOLVER.MIN_EPOCHS,
            max_epochs=cfg.SOLVER.MAX_EPOCHS,
            checkpoint_callback=checkpoint_callback,
            # resume_from_checkpoint=last_checkpoint_file,
            gpus=args.gpus,
            logger=False,  # logger,
            # weights_summary='full',  
            fast_dev_run=False,  # True,
        )

        trainer.fit(model)
        results.update(
            is_validation=True,
            method_name=cfg.DAN.METHOD,
            seed=seed,
            metric_values=trainer.callback_metrics,
        )
        # test scores
        trainer.test()
        results.update(
            is_validation=False,
            method_name=cfg.DAN.METHOD,
            seed=seed,
            metric_values=trainer.callback_metrics,
        )
        results.to_csv(test_csv_file)
        results.print_scores(cfg.DAN.METHOD)


if __name__ == '__main__':
    main()
