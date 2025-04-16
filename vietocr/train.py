import argparse

from vietocr.model.trainer import Trainer
from vietocr.tool.config import Cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=False, help="see example at ")
    parser.add_argument("--checkpoint", required=False, help="your checkpoint")

    args = parser.parse_args()
    config = Cfg.load_config_from_name('vgg_transformer')
    dataset_params = {
        'name': 'hw',
        'data_root': './data_line/',
        'train_annotation': 'train_line_annotation.txt',
        'valid_annotation': 'test_line_annotation.txt'
    }
    params = {
        'print_every': 200,
        'valid_every': 15 * 200,
        'iters': 20000,
        'checkpoint': './checkpoint/transformerocr_checkpoint.pth',
        'export': './weights/transformerocr.pth',
        'metrics': 10000
    }
    config['trainer'].update(params)
    config['dataset'].update(dataset_params)
    config['device'] = 'cpu'
    trainer = Trainer(config, pretrained=True)

    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)

    trainer.train()


if __name__ == "__main__":
    main()