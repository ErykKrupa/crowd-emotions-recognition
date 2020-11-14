from config.config import Config
from data.data_preprocessor import DataPreprocessor
from data.data_set import DataSet

print("Hello!")

print("Train set length: " + str(len(DataPreprocessor.get_all_pictures_metadata(DataSet.TRAIN))))
print("Test set length: " + str(len(DataPreprocessor.get_all_pictures_metadata(DataSet.TEST))))
print("Validation set length: " + str(len(DataPreprocessor.get_all_pictures_metadata(DataSet.VALIDATION))))

# todo model
# history = model.fit_generator(
#     DataPreprocessor.get_train_generator(),
#     steps_per_epoch=Config.STEPS_PER_EPOCH,
#     epochs=Config.EPOCHS,
#     validation_data=DataPreprocessor.get_validation_generator(),
#     validation_steps=Config.VALIDATION_STEPS
# )
