from keras.applications import *

# files
DATA_DIRECTORY = 'data'
MODELS_DIRECTORY = 'models'
EXTRACTED_DATA_CACHE_DIRECTORY = 'extracted_cache'

# preprocessing
PICTURE_SIZE = (150, 150)
ROTATION_RANGE = 30
WIDTH_SHIFT_RANGE = 0.1
HEIGHT_SHIFT_RANGE = 0.1
SHEAR_RANGE = 0.1
ZOOM_RANGE = 0.1
HORIZONTAL_FLIP = True
VERTICAL_FLIP = False
FILL_MODE = 'nearest'

# model construction
INPUT_SHAPE = PICTURE_SIZE + (3,)
KERNEL_SIZE = (3, 3)
POOL_SIZE = (2, 2)
ACTIVATION = 'relu'
OPTIMIZER_LEARNING_RATE = 2e-5
DROPOUT_RATE = 0.5
PRETRAINED_MODEL = VGG16

# model launching
USE_PRETRAINED_MODEL = True
BATCH_SIZE = 32
EPOCHS = 10

# learning results visualization
TRAIN_LINE_STYLE = 'b-'
VALIDATION_LINE_STYLE = 'g-'
