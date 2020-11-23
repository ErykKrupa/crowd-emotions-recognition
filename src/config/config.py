from keras.applications import *

# files
DATA_DIRECTORY = 'data'
MODELS_DIRECTORY = 'models'
RESULTS_DIRECTORY = 'results'
EXTRACTED_DATA_CACHE_DIRECTORY = 'extracted_cache'

# preprocessing
PICTURE_SIZE = 224  # 128, 160, 192, 224

# augmentation
ROTATION_RANGE = 30
WIDTH_SHIFT_RANGE = 0.1
HEIGHT_SHIFT_RANGE = 0.1
SHEAR_RANGE = 0.1
ZOOM_RANGE = 0.1
HORIZONTAL_FLIP = True
VERTICAL_FLIP = False
FILL_MODE = 'nearest'

# model construction
KERNEL_SIZE = 3
POOL_SIZE = 2
ACTIVATION = 'relu'
OPTIMIZER_LEARNING_RATE = 2e-5
DROPOUT_RATE = 0.5

# pretrained model construction
PRETRAINED_MODEL = VGG16

# model launching
USE_PRETRAINED_MODEL = True
BATCH_SIZE = 32
EPOCHS = 10

# learning results visualization
TRAIN_LINE_STYLE = 'b-'
VALIDATION_LINE_STYLE = 'g-'
