from keras import callbacks
from keras.optimizers import Adam, SGD
import numpy as np

import os

import model
import data
import losses

path = "./data/"

NO_OF_TRAINING_IMAGES = len(os.listdir(path + 'train/train_frames/'))
NO_OF_VAL_IMAGES = len(os.listdir(path + 'val/val_frames/'))

NO_OF_EPOCHS = 8
BATCH_SIZE = 4
weights_path = path
INPUT_SHAPE = (256, 256, 3)
LR = 1e-1
num = 4

processor = data.DataProcessor(INPUT_SHAPE[0], INPUT_SHAPE[1], batch_size=BATCH_SIZE)

rgb_to_id = processor.get_codes()[1]

CLASSES = len(rgb_to_id)

print("loading data")
print('-' * 30)

train_gen = processor.get_generator('train', INPUT_SHAPE)
val_gen = processor.get_generator('val', INPUT_SHAPE)
test_gen = processor.get_generator('test', INPUT_SHAPE)

unet_model = model.unet(INPUT_SHAPE, CLASSES)
# opt = Adam(lr=LR, beta_1=0.9, beta_2=1-1e-3, epsilon=1e-08)
opt = SGD(learning_rate=LR)
unet_model.compile(opt, loss=[losses.focal_loss(alpha=.25, gamma=2)])

# unet_model.summary()

print("get unet")
print('-' * 30)

checkpoint = callbacks.ModelCheckpoint(weights_path, monitor=losses.focal_loss,
                             verbose=1, save_best_only=True, mode='min')
csv_logger = callbacks.CSVLogger('./results/log.out', append=True, separator=';')
earlystopping = callbacks.EarlyStopping(monitor=losses.focal_loss, verbose=1,
                              min_delta=0.01, patience=3, mode='min')
callbacks_list = [checkpoint, csv_logger, earlystopping]

print("fitting model...")
print('-' * 30)

results = unet_model.fit_generator(train_gen,
                                   epochs=NO_OF_EPOCHS,
                                   steps_per_epoch=(NO_OF_TRAINING_IMAGES//BATCH_SIZE),
                                   validation_data=val_gen,
                                   validation_steps=(NO_OF_VAL_IMAGES//BATCH_SIZE),
                                   callbacks=callbacks_list)

print(results)
print("save history")
print('-' * 30)
np.save(f'./results/unet_history{num:03}.npy', results.history)

# load
# history = np.load('./results/unet_history.npy', allow_pickle='TRUE').item()

print("save model")
print('-' * 30)
unet_model.save(f'./results/Model{num:03}.h5')
test_res = unet_model.evaluate_generator(test_gen, steps=30)
with open('./results/results.txt', 'a+') as f:
    f.write(f"{num:03} {BATCH_SIZE} {NO_OF_EPOCHS} {LR} {test_res}")
