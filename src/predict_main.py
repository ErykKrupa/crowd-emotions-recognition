from copy import deepcopy
from os import listdir
from os.path import isdir, isfile
from sys import stderr

from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img
import argparse
import numpy as np

from config.config import Config
from utils.logger import log_info

parser = argparse.ArgumentParser()
parser.add_argument('-m',
                    '--model',
                    type=str,
                    required=Config.get('default_model_path') is None,
                    default=Config.get('default_model_path'),
                    help='path to the saved model in .h5 format. Set this path using "default_model_path" field in '
                         'config.json to avoid passing this parameter every time')
parser.add_argument('paths',
                    type=str,
                    nargs='+',
                    help='paths to the images for prediction in .jpg format, '
                         'or to the directories containing those images')
args = parser.parse_args()

images = []
paths = deepcopy(args.paths)
first_not_found = True
for path in paths:
    if isdir(path):
        paths.extend(path + '/' + f for f in listdir(path))
    elif isfile(path) and path.endswith('.jpg'):
        images.append(path)
    else:
        if first_not_found:
            first_not_found = False
            print("These files don't exist or are in the wrong format: ", file=stderr)
        print(path, file=stderr)

if not images:
    print('No correct paths have been found', file=stderr)
    exit(1)

log_info('Loading model...')
model = load_model(args.model)
log_info(f'Model {args.model} loaded')

log_info('Preparing data...')
input_arr = [img_to_array(load_img(i, target_size=model.inputs[0].shape[1:3])) for i in images]
input_arr = np.asarray(input_arr)
log_info('Data prepared')

log_info('Recognizing emotions...')
predictions = model.predict(input_arr)
for prediction, name in zip(predictions, images):
    print()
    print(f'Prediction for: {name}')
    print(f'Positive: {prediction[2] * 100:>6.2f}%')
    print(f'Neutral:  {prediction[1] * 100:>6.2f}%')
    print(f'Negative: {prediction[0] * 100:>6.2f}%')
