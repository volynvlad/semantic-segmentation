from keras import callbacks
from keras.optimizers import Adam

import os

import model
import data

path="./cityscapes_data/"

NO_OF_TRAINING_IMAGES = len(os.listdir(path + 'train_frames/'))
NO_OF_VAL_IMAGES = len(os.listdir(path + 'val_frames/'))

NO_OF_EPOCHS = 30
BATCH_SIZE = 8
weights_path = path
CLASSES = 13
INPUT_SHAPE = (256, 256, 3)

processor = data.DataProcessor(256, 256)

print("loading data")

train_X = processor.get_frame_data('train')
val_X = processor.get_frame_data('val')
test_X = processor.get_frame_data('test')

train_y = processor.get_mask_data('train')
val_y = processor.get_mask_data('val')
test_y = processor.get_mask_data('test')

# train_X = train_X / 255.0
# train_y = train_y / 255.0
# val_X = val_X / 255.0
# val_y = val_y / 255.0
# test_X = test_X / 255.0
# test_y = test_y / 255.0


unet_model = model.unet(INPUT_SHAPE, CLASSES)
print("get unet")

checkpoint = callbacks.ModelCheckpoint(weights_path, monitor='categorical_crossentropy',
                             verbose=1, save_best_only=True, mode='min')
csv_logger = callbacks.CSVLogger('./log.out', append=True, separator=';')
earlystopping = callbacks.EarlyStopping(monitor = 'categorical_crossentropy', verbose = 1,
                              min_delta = 0.01, patience = 3, mode = 'min')
callbacks_list = [checkpoint, csv_logger, earlystopping]

# results = m.fit_generator(train_gen,
#                           steps_per_epoch=(NO_OF_TRAINING_IMAGES//BATCH_SIZE),
#                           epochs=NO_OF_EPOCHS,
#                           validation_data=val_gen,
#                           validation_steps=(NO_OF_VAL_IMAGES//BATCH_SIZE),
#                           callbacks=callbacks_list)

print("fitting model...")
results = unet_model.fit(train_X, train_y,
                steps_per_epoch=NO_OF_TRAINING_IMAGES // BATCH_SIZE,
                epochs=NO_OF_EPOCHS,
                validation_data=(val_X, val_y),
                validation_steps=(NO_OF_TRAINING_IMAGES // BATCH_SIZE),
                callbacks=callbacks_list)

print(results)
with open("history.txt") as f:
    f.write("reseults = " + results)
print(f"score = {unet_model.score(test_X, test_y)}")
print("save model")
m.save('Model.h5')
