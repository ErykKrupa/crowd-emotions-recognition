import argparse
from copy import deepcopy
from os import listdir
from os.path import isdir, isfile
from sys import stderr

import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img

from config.config import Config
from utils.logger import log_info

parser = argparse.ArgumentParser()
parser.add_argument('-m',
                    '--model',
                    type=str,
                    required=Config.get('default_model_path') is None,
                    default=Config.get('default_model_path'),
                    help='path to the saved model in .h5 format. Set path '
                         'using "default_model_path" in config.json to '
                         'avoid passing this parameter every time')
parser.add_argument('paths',
                    type=str,
                    nargs='+',
                    help='paths to the images for prediction in .jpg format, '
                         'or to the directories containing those images')
parser.add_argument('-v',
                    '--verbose',
                    action="store_true",
                    help='show detailed predictions results')
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
            print("These files don't exist or are in the wrong formats: ", file=stderr)
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

if args.verbose:
    for prediction, name in zip(predictions, images):
        print()
        print(f'Prediction for: {name}')
        print(f'Positive: {prediction[2] * 100:>6.2f}%')
        print(f'Neutral:  {prediction[1] * 100:>6.2f}%')
        print(f'Negative: {prediction[0] * 100:>6.2f}%')
else:
    print(f'Predicted emotions: ')
    max_name_length = max(map(lambda a: len(a), images))
    max_name_length = min(max_name_length, 85)
    max_name_length = max(max_name_length, 30)
    for prediction, name in zip(predictions, images):
        predicted_emotion, higher = ('positive', prediction[2]) \
            if prediction[2] > prediction[0] \
            else ('negative', prediction[0])
        if prediction[1] > higher:
            predicted_emotion = 'neutral'
        print(f'{name:.<{max_name_length}}{predicted_emotion:.>15}')
