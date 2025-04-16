import argparse

from vietocr.model.trainer import Trainer
from vietocr.tool.config import Cfg
from vietocr.tool.predictor import Predictor

import urllib.request
import zipfile
import os
import random
from PIL import Image
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="../config/base.yml",
        help="Path to config yml file (default: base.yml)"
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to checkpoint file (optional)"
    )
    args = parser.parse_args()
    config = Cfg.load_config_from_name('vgg_transformer')
    config['cnn']['pretrained'] = False
    config['device'] = 'cpu'
    detector = Predictor(config)


    img_path = './sample/img_21.png'
    img = Image.open(img_path)

    # 5. Hiển thị ảnh
    plt.imshow(img)
    plt.axis('off')
    plt.title('Ảnh mẫu')
    plt.show()

    # 6. Thực hiện nhận diện chữ viết
    result = detector.predict(img)
    print("Kết quả OCR:", result)

    # config = Cfg.load_config_from_file(args.config)
    #
    # trainer = Trainer(config)
    #
    # if args.checkpoint:
    #     trainer.load_checkpoint(args.checkpoint)
    #
    # trainer.train()


if __name__ == "__main__":
    main()
