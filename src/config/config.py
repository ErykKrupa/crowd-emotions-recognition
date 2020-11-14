class Config:
    DATA_DIRECTORY = 'data/'
    PICTURE_SIZE = 150
    BATCH_SIZE = 20
    STEPS_PER_EPOCH = 100  # should BATCH_SIZE * STEPS_PER_EPOCH == data size ?
    EPOCHS = 30
    VALIDATION_STEPS = 50
