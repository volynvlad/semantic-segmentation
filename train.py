from keras import callbacks
from keras.optimizers import Adam

import os

import model
import data
import dice_loss

path="./data/"

NO_OF_TRAINING_IMAGES = len(os.listdir(path + 'train_frames/'))
NO_OF_VAL_IMAGES = len(os.listdir(path + 'val_frames/'))

NO_OF_EPOCHS = 30
BATCH_SIZE = 8
weights_path = path
INPUT_SHAPE = (256, 256, 3)

processor = data.DataProcessor(256, 256)

rgb_to_id = processor.get_codes()[1]

CLASSES = len(rgb_to_id)

print("loading data")
print('-' * 30)

train_X = processor.get_frame_data('train')
val_X = processor.get_frame_data('val')
test_X = processor.get_frame_data('test')

train_y = processor.get_mask_data('train')
val_y = processor.get_mask_data('val')
test_y = processor.get_mask_data('test')


print(f"train_X.shape={train_X.shape}")
print(f"val_X.shape={val_X.shape}")
print(f"test_X.shape={test_X.shape}")

print(f"train_y.shape={train_y.shape}")
print(f"val_y.shape={val_y.shape}")
print(f"test_y.shape={test_y.shape}")

unet_model = model.unet(INPUT_SHAPE, CLASSES)
opt = Adam(lr=1e-5, beta_1=0.9, beta_2=1-1e-3, epsilon=1e-08)
unet_model.compile(opt, loss=dice_loss.dice_coef_loss, metrics=[dice_loss.dice_coef])
# unet_model.summary()
print("get unet")
print('-' * 30)

checkpoint = callbacks.ModelCheckpoint(weights_path, monitor=dice_loss.dice_coef_loss,
                             verbose=1, save_best_only=True, mode='min')
csv_logger = callbacks.CSVLogger('./log.out', append=True, separator=';')
earlystopping = callbacks.EarlyStopping(monitor=dice_loss.dice_coef_loss, verbose=1,
                              min_delta=0.01, patience=3, mode='min')
callbacks_list = [checkpoint, csv_logger, earlystopping]

print("fitting model...")
print('-' * 30)
results = unet_model.fit(train_X, train_y,
                steps_per_epoch=NO_OF_TRAINING_IMAGES // BATCH_SIZE,
                epochs=NO_OF_EPOCHS,
                validation_data=(val_X, val_y),
                validation_steps=(NO_OF_TRAINING_IMAGES // BATCH_SIZE),
                callbacks=callbacks_list)

print(results)
print("save history")
np.save('unet_history.npy', history.history)
# load
# history = np.load('unet_history.npy', allow_pickle='TRUE').item()

print(f"score = {unet_model.score(test_X, test_y)}")
print("save model")
unet_model.save('Model.h5')
