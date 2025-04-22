from tool.config import Cfg
from tool.predictor import Predictor

from PIL import Image
import matplotlib.pyplot as plt

def main():
    # Load configuration for the OCR model
    # config = Cfg.load_config_from_name('vgg_transformer')
    config = Cfg.load_config_from_file('./config/vgg-transformer.yml')
    
    config['device'] = 'cpu'
    print('config', config)
    
    # Initialize the OCR detector
    detector = Predictor(config)

    # Load the image
    img_path = './sample/img_21.png'
    img = Image.open(img_path)


    # Perform OCR prediction
    result = detector.predict(img)
    print("Kết quả OCR:", result)

# Display the image
    plt.imshow(img)
    plt.axis('off')
    plt.title('PREDICT: ' + result)
    plt.show()

if __name__ == "__main__":
    main()