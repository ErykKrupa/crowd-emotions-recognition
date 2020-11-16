from math import ceil

from config.config import EPOCHS, TRAIN_BATCH_SIZE, VALIDATION_BATCH_SIZE
from data.data_preprocessor import get_amount_of_pictures, get_train_generator, get_validation_generator
from data.data_set import DataSet
from model.model import prepare_model

model = prepare_model()
model.summary()
# todo launching model?
history = model.fit_generator(
    get_train_generator(),
    steps_per_epoch=ceil(get_amount_of_pictures(DataSet.TRAIN) / TRAIN_BATCH_SIZE),
    epochs=EPOCHS,
    validation_data=get_validation_generator(),
    validation_steps=ceil(get_amount_of_pictures(DataSet.VALIDATION) / VALIDATION_BATCH_SIZE)
)
model.save('model_v1.h5')
