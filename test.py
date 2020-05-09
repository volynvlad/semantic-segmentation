import numpy as np
import cv2

import model
import data

proc = data.DataProcessor(256, 256)

id_to_rgb, rgb_to_id = proc.get_codes()

unet_model = model.unet((256, 256, 3), len(id_to_rgb))
unet_model.load_weights('results/Model001.h5')
image = cv2.imread('./data/test/test_frames/frame_0000.png')
image = image.reshape((1, 256, 256, 3))
pred_image = unet_model.predict(image)
pred_image = pred_image.reshape((256, 256, len(id_to_rgb)))

res_image = np.zeros((256, 256, 3))
for x in range(256):
    for y in range(256):
        res_image[x, y] = id_to_g[np.argmax(pred_image[x, y])]

print(res_image)
np.save('image.npy', res_image)

