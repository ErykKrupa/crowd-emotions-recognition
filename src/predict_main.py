from os import environ

from config.config import Config

environ['TF_CPP_MIN_LOG_LEVEL'] = '0' if Config.get('tensorflow_log') else '3'

import argparse
import json
from copy import deepcopy
from os import listdir
from os.path import isdir, isfile
from sys import stderr

import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img

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
parser.add_argument('-o',
                    '--output',
                    type=str,
                    required=False,
                    help='instead of showing results on standard output, '
                         'write it to .json file whose format depends '
                         'on whether --verbose is specified')
parser.add_argument('-v',
                    '--verbose',
                    action='store_true',
                    help='show detailed predictions results, or write more '
                         'information to file when --output option is specified')
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
model = load_model(args.model, compile=False)
log_info(f'Model {args.model} loaded')

log_info('Preparing data...')
input_arr = [img_to_array(load_img(i, target_size=model.inputs[0].shape[1:3])) for i in images]
input_arr = np.asarray(input_arr)
log_info('Data prepared')

log_info('Recognizing emotions...')
predictions = model.predict(input_arr)

results = {}
if args.verbose:
    results = {name: {
        'positive': round(prediction[2] * 100, 2),
        'neutral': round(prediction[1] * 100, 2),
        'negative': round(prediction[0] * 100, 2),
    } for prediction, name in zip(predictions, images)}
else:
    for prediction, name in zip(predictions, images):
        predicted_emotion, higher = ('positive', prediction[2]) \
            if prediction[2] > prediction[0] \
            else ('negative', prediction[0])
        if prediction[1] > higher:
            predicted_emotion = 'neutral'
        results[name] = predicted_emotion

if args.output:
    with open(args.output, 'w') as file:
        json.dump(results, file, indent=4)
    log_info(f'Results saved in {args.output} file')
    exit(0)

if args.verbose:
    for name, result in results.items():
        print()
        print(f'Prediction for: {name}')
        print(f'Positive: {dict(result)["positive"]:>6.2f}%')
        print(f'Neutral:  {dict(result)["neutral"]: >6.2f}%')
        print(f'Negative: {dict(result)["negative"]:>6.2f}%')
else:
    print(f'Predicted emotions: ')
    max_name_length = max(map(lambda a: len(a), images))
    max_name_length = min(max_name_length, 85)
    max_name_length = max(max_name_length, 30)
    for name, emotion in results.items():
        print(f'{name:.<{max_name_length}}{emotion:.>15}')
