import numpy as np
import cv2

import model
import data

proc = data.DataProcessor(256, 256)

id_to_rgb, rgb_to_id = proc.get_codes()

unet_model = model.unet((256, 256, 3), len(id_to_rgb))
unet_model.load_weights('results/Model003.h5')
image = cv2.imread('./data/train/train_frames/frame_0000.png')
image = image.reshape((1, 256, 256, 3))
pred_image = unet_model.predict(image)
pred_image = pred_image.reshape((256, 256, len(id_to_rgb)))

res_image = np.zeros((256, 256, 3))
for x in range(256):
    for y in range(256):
        res_image[x, y] = np.array(id_to_rgb[np.argmax(pred_image[x, y])])

cv2.imwrite('result_train0000.png', res_image)

