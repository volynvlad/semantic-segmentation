from keras.preprocessing.image import ImageDataGenerator
import cv2

import numpy as np

import os


class DataProcessor(object):
    def __init__(self, out_rows, out_cols,
                 path="./cityscapes_data/",
                 img_type="jpg", batch_size=8):
        self.out_rows = out_rows
        self.out_cols = out_cols
        self.path = path
        self.img_type = img_type
        self.batch_size = batch_size

    def divide_to_frame_mask(self):
        train_names = os.listdir(self.path + 'train/')
        val_names = os.listdir(self.path + 'val/')
        train_size = int(len(train_names) * 17 / 18)

        for i in range(train_size):
            image = cv2.imread(self.path + 'train/' + train_names[i])
            image_frame = image[:, :self.out_cols]
            image_mask = image[:, self.out_cols:]
            cv2.imwrite(self.path + "train_frames/frame_{:04}.{}".format(i, self.img_type), image_frame)
            cv2.imwrite(self.path + "train_masks/mask_{:04}.{}".format(i, self.img_type), image_mask)

        for i in range(train_size, len(train_names)):
            image = cv2.imread(self.path + 'train/' + train_names[i])
            image_frame = image[:, :self.out_cols]
            image_mask = image[:, self.out_cols:]
            cv2.imwrite(self.path + "test_frames/frame_{:04}.{}".format(i - train_size, self.img_type), image_frame)
            cv2.imwrite(self.path + "test_masks/mask_{:04}.{}".format(i - train_size, self.img_type), image_mask)

        for i in range(len(val_names)):
            image = cv2.imread(self.path + 'val/' + val_names[i])
            image_frame = image[:, :self.out_cols]
            image_mask = image[:, self.out_cols:]
            cv2.imwrite(self.path + "val_frames/frame_{:04}.{}".format(i, self.img_type), image_frame)
            cv2.imwrite(self.path + "val_masks/mask_{:04}.{}".format(i, self.img_type), image_mask)

    def data_generation(self):
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)
        val_datagen = ImageDataGenerator(rescale=1./255)

        train_image_generator = train_datagen.flow_from_directory(self.path + "train_frames",
                                                                  batch_size=self.batch_size)
        train_mask_generator = train_datagen.flow_from_directory(self.path + "train_masks",
                                                                  batch_size=self.batch_size)
        val_image_generator = val_datagen.flow_from_directory(self.path + "val_frames",
                                                                  batch_size=self.batch_size)
        val_mask_generator = val_datagen.flow_from_directory(self.path + "val_masks",
                                                                  batch_size=self.batch_size)
        train_generator = list((train_image_generator, train_mask_generator))
        val_generator = list((val_image_generator, val_mask_generator))

        return train_generator, val_generator

    def get_frame_data(self, name):
        print(f"load {name} frame data...")
        print('-' * 30)
        path_frames = self.path + name + "_frames/"

        frame_names = os.listdir(path_frames)

        X = []

        for frame_name in frame_names:
            image = cv2.imread(path_frames + frame_name)
            X.append(image)

        return np.array(X)

    def get_mask_data(self, name):
        print(f"load {name} mask data...")
        print('-' * 30)
        path_masks = self.path + name + "_masks/"

        masks_names = os.listdir(path_masks)

        y = []

        for masks_name in masks_names:
            image = cv2.imread(path_masks + masks_name)
            # TODO convert image to n images, n - number of classes on the images
            y.append(image)

        return np.array(y)


if __name__ == "__main__":
    processor = DataProcessor(256, 256)

    processor.divide_to_frame_mask()

