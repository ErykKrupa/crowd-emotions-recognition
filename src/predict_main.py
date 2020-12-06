from os import environ

from config.config import predict_config as config

environ['TF_CPP_MIN_LOG_LEVEL'] = '0' if config.get('tensorflow_log') else '3'

import argparse
import json
from copy import deepcopy
from os import listdir
from os.path import isdir, isfile
from sys import stderr

import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img

from utils.logger import predict_log as log

parser = argparse.ArgumentParser()
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
            print("These files don't exist or are in the wrong formats: ", file=stderr)
        print(path, file=stderr)

if not images:
    print('No correct paths have been found', file=stderr)
    exit(1)

log('Loading model...')
model = load_model(config.get('model_path'), compile=False)
log(f'Model {config.get("model_path")} loaded')

log('Preparing data...')
input_arr = [img_to_array(load_img(i, target_size=model.inputs[0].shape[1:3])) for i in images]
input_arr = np.asarray(input_arr)
log('Data prepared')

log('Recognizing emotions...')
predictions = model.predict(input_arr)

results = {}
if config.get('verbose'):
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

if config.get('output_file'):
    file_name = config.get('output_file') + '.json'
    with open(file_name, 'w') as file:
        json.dump(results, file, indent=4)
    log(f'Results saved in {file_name} file')
    exit(0)

if config.get('verbose'):
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
