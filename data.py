from keras.preprocessing.image import ImageDataGenerator
import cv2

import numpy as np

import os


class DataProcessor(object):
    def __init__(self, out_rows, out_cols,
                 path='./data/',
                 frame_path="./data/leftImg8bit/",
                 mask_path="./data/gtFine/",
                 img_type="png", batch_size=8):
        self.out_rows = out_rows
        self.out_cols = out_cols
        self.path = path
        self.frame_path = frame_path
        self.mask_path = mask_path
        self.img_type = img_type
        self.batch_size = batch_size


    def reconstruct_folders(self):
        train_size = int(sum(len(os.listdir(self.frame_path + 'train/' + name))
                             for name in os.listdir(self.frame_path + 'train/')) * 17 / 18)
        os.mkdir(self.path + "frames")
        os.mkdir(self.path + "test_frames")
        os.mkdir(self.path + "test_masks")
        for name in ['train', 'val']:
            os.mkdir(self.path + name + "_frames")
            os.mkdir(self.path + name + "_masks")
            count = 0
            # same city names for frames and masks
            city_names = os.listdir(self.frame_path + name)
            for city_name in city_names:
                image_names = os.listdir(self.frame_path + name + '/' + city_name)

                for image_name in image_names:
                    frame_image = cv2.imread(self.frame_path + name + '/' + city_name + '/' + image_name)
                    mask_image = cv2.imread(self.mask_path + name + '/' + city_name + '/' + image_name.replace('leftImg8bit', 'gtFine_color'))
                    is_test = count >= train_size
                    cv2.imwrite(self.path + f'{"test" if is_test else name}_frames/frame_{count - train_size if is_test else count:04}.{self.img_type}',
                        cv2.resize(frame_image, (self.out_rows, self.out_cols), interpolation=cv2.INTER_NEAREST))
                    cv2.imwrite(self.path + f'{"test" if is_test else name}_masks/mask_{count - train_size if is_test else count:04}.{self.img_type}',
                        cv2.resize(mask_image, (self.out_rows, self.out_cols), interpolation=cv2.INTER_NEAREST))
                    count += 1

        city_names = os.listdir(self.frame_path + "test")
        count = 0
        for city_name in city_names:
            image_names = os.listdir(self.frame_path + 'test/' + city_name)
            for image_name in image_names:
                frame_image = cv2.imread(self.frame_path + 'test/' + city_name + '/' + image_name)
                cv2.imwrite(self.path + f'frames/frame_{count:04}.{self.img_type}',
                    cv2.resize(frame_image, (self.out_rows, self.out_cols), interpolation=cv2.INTER_NEAREST))
                count += 1



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
#             y.append(self.mask_to_onehot(image, classes, rgbtoid))

        return np.array(y)

    def mask_to_onehot(self, mask, classes, rgbtoid):
        onehot_mask = np.zeros((mask.shape[0], mask.shape[1], classes))

        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                onehot_mask[i, j, rdbtoid[mask[i, j]]] = 1

        return onehot_mask


if __name__ == "__main__":
    processor = DataProcessor(256, 256)
    
    processor.reconstruct_folders()

