DATA_DIRECTORY = 'data/'

# preprocessing
PICTURE_SIZE = (150, 150)
BATCH_SIZE = 20
STEPS_PER_EPOCH = 100  # should BATCH_SIZE * STEPS_PER_EPOCH == data size ?
EPOCHS = 30
VALIDATION_STEPS = 50

# model
INPUT_PICTURE_SHAPE = PICTURE_SIZE + (3,)
KERNEL_SIZE = (3, 3)
POOL_SIZE = (2, 2)