from data.data_preprocessor import get_all_pictures_metadata
from data.data_set import DataSet
# from model.model import prepare_model

print('Hello!')

print('Train set length: ' + str(len(get_all_pictures_metadata(DataSet.TRAIN))))
print('Test set length: ' + str(len(get_all_pictures_metadata(DataSet.TEST))))
print('Validation set length: ' + str(len(get_all_pictures_metadata(DataSet.VALIDATION))))

# todo model
# prepare_model().summary()
#
# history = model.fit_generator(
#     get_train_generator(),
#     steps_per_epoch=STEPS_PER_EPOCH,
#     epochs=EPOCHS,
#     validation_data=get_validation_generator(),
#     validation_steps=VALIDATION_STEPS
# )
