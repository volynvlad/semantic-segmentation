import cv2
import numpy as np

import os
import random


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
        print("reconstruct folders")
        print('-' * 30)
        train_size = int(sum(len(os.listdir(self.frame_path + 'train/' + name))
                             for name in os.listdir(self.frame_path + 'train/')) * 17 / 18)

        os.mkdir(self.path + "train")
        os.mkdir(self.path + "val")
        os.mkdir(self.path + "test")
        os.mkdir(self.path + "frames")
        os.mkdir(self.path + "test/test_frames")
        os.mkdir(self.path + "test/test_masks")
        for name in ['train', 'val']:
            os.mkdir(self.path + name + "/" + name + "_frames")
            os.mkdir(self.path + name + "/" + name + "_masks")
            count = 0
            # same city names for frames and masks
            city_names = os.listdir(self.frame_path + name)
            for city_name in city_names:
                image_names = os.listdir(self.frame_path + name + '/' + city_name)

                for image_name in image_names:
                    frame_image = cv2.imread(self.frame_path + name + '/' + city_name + '/' + image_name)
                    mask_image = cv2.imread(self.mask_path + name + '/' + city_name + '/' + image_name.replace('leftImg8bit', 'gtFine_color'))
                    is_test = count >= train_size
                    cv2.imwrite(self.path + f'{"test" if is_test else name}/{"test" if is_test else name}_frames/frame_{count - train_size if is_test else count:04}.{self.img_type}',
                                cv2.resize(frame_image, (self.out_rows, self.out_cols), interpolation=cv2.INTER_NEAREST))
                    cv2.imwrite(self.path + f'{"test" if is_test else name}/{"test" if is_test else name}_masks/mask_{count - train_size if is_test else count:04}.{self.img_type}',
                                cv2.resize(mask_image, (self.out_rows, self.out_cols), interpolation=cv2.INTER_NEAREST))
                    count += 1

        city_names = os.listdir(self.frame_path + "test")
        count = 0
        for city_name in city_names:
            image_names = os.listdir(self.frame_path + 'test/' + city_name)
            for image_name in image_names:
                frame_image = cv2.imread(self.frame_path + 'test/' + city_name + '/' + image_name)
                cv2.imwrite(self.path + 'test/' + f'frames/frame_{count:04}.{self.img_type}',
                            cv2.resize(frame_image, (self.out_rows, self.out_cols), interpolation=cv2.INTER_NEAREST))
                count += 1

    @staticmethod
    def get_codes():
        id_to_rgb = {}
        rgb_to_id = {}
        codes_path = './id_to_rgb.txt'
        codes = open(codes_path, 'r')
        lines = codes.readlines()
        for line in lines:
            numbers = [int(x) for x in line.replace('\n', '').split(',')]
            rgb_to_id[tuple(numbers[1:])] = numbers[0]
            id_to_rgb[numbers[0]] = tuple(numbers[1:])

        return id_to_rgb, rgb_to_id

    def get_generator(self, name, shape):
        print(f"get {name} generator")
        print('-' * 30)

        rgb_to_id = self.get_codes()[1]

        classes = len(rgb_to_id)

        c = 0
        frames_folder = f"{self.path}{name}/{name}_frames"
        masks_folder = f"{self.path}{name}/{name}_masks"

        n = os.listdir(frames_folder)
        random.shuffle(n)

        while True:
            frame = np.zeros((self.batch_size, *shape)).astype('float')
            mask = np.zeros((self.batch_size, shape[0], shape[1], classes))

            for i in range(c, c + self.batch_size):
                train_frame = cv2.imread(f"{frames_folder}/{n[i]}") / 255.
                train_frame = cv2.resize(train_frame, (shape[0], shape[1]))

                frame[i - c] = train_frame

                train_mask = cv2.imread(f"{masks_folder}/{n[i].replace('frame', 'mask')}")
                #                 ret_train_mask = np.zeros((shape[0], shape[1], classes))

                for x in range(shape[0]):
                    for y in range(shape[1]):
                        mask[i - c, x, y, rgb_to_id[tuple(train_mask[x, y])]] = 1

#                 mask[i - c] = train_mask

            c += self.batch_size
            if c + self.batch_size >= len(os.listdir(frames_folder)):
                c = 0
                random.shuffle(n)

            yield frame, mask


if __name__ == "__main__":
    processor = DataProcessor(256, 256)
    processor.reconstruct_folders()
