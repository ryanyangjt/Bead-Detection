import cv2
import os
import numpy as np
import tensorflow as tf
import gc
import itertools
import time


def batch_predict(axis):
    axis = np.asarray(axis, dtype=float)
    sc = model.predict(axis)
    return sc


# Create directory for saving detection results
detected_dir = "./detected_images/"
if os.path.exists(detected_dir):
    print("[INFO] The detected_images director already exist!")
    pass
else:
    os.mkdir(detected_dir)
    print("[INFO] Creating detected_images director!")
# Loading model:
model = tf.keras.models.load_model("bead_detection_model.h5")
model.summary()


blur_img_path = "./Blur_Images/"
blur_imgs = os.listdir(blur_img_path)

# b_img = blur_imgs[0]
for b_img in blur_imgs:
    image_path = blur_img_path + b_img
    image = cv2.imread(image_path)
    print("[INFO] Detecting Image: " + b_img)
    print("[INFO] Image Size: ", image.shape)
    tmp = image
    image = image/255
    count = 0
    stepSize = 12
    (w_width, w_height) = (80, 80)  # window size
    imgs_window = []
    axis_window = []

    for y in range(0, image.shape[0] - w_height, stepSize):
        for x in range(0, image.shape[1] - w_width, stepSize):
            img_read = image[y:y + w_height, x:x + w_width, :]
            imgs_window.append(img_read)
            axis_window.append([y, x, y + w_height, x + w_width])
            count += 1
    print("[INFO] Total number of ROIs: ", count)

    b_num = 5000    # this parameter is to control the number of prediction for each batch.
    score_info = []
    total_num = len(imgs_window)
    total_run = int(total_num/b_num)
    start_time = time.time()
    for n in range(total_run):
        print("[INFO] Predicting progress => [" + str((n+1)*b_num) + "/" + str(total_num) + "] ")
        temp_window = imgs_window[n*b_num:(n+1)*b_num]
        score_info.append(batch_predict(temp_window))
        gc.collect()    # release the memory to avoid out of memory.

    temp_w = np.asarray(imgs_window[total_run*b_num::], dtype=float)
    print("[INFO] Predicting progress => [" + str(total_num) + "/" + str(total_num) + "] ")
    score_info.append(model.predict(temp_w))
    end_time = time.time()
    print("[INFO] classifying ROIs took {:.5f} seconds".format(end_time - start_time))
    score_info = list(itertools.chain.from_iterable(score_info))

    # keep the bead samples
    score_info_checked = []
    axis_window_checked = []
    for i in range(len(score_info)):
        if 0.00001 < score_info[i] < 0.25:
            score_info_checked.append(float(1 - score_info[i]))     # 0: bead, 1: dust
            axis_window_checked.append(axis_window[i])

    # show all bead windows
    axis_window_checked = np.asarray(axis_window_checked, dtype=float)
    score_info_checked = np.asarray(score_info_checked, dtype=float)
    for y, x, y2, x2 in axis_window_checked:
        cv2.rectangle(tmp, (int(x), int(y)), (int(x2), int(y2)), (255, 0, 0), 2)

    # non_maximum_suppression
    selected_indices = tf.image.non_max_suppression(
        boxes=axis_window_checked, scores=score_info_checked, iou_threshold=0.5, max_output_size=200)
    # draw
    selected_boxes = tf.gather(axis_window_checked, selected_indices)
    select_img = cv2.imread(image_path)
    for y, x, y2, x2 in selected_boxes:
        cv2.rectangle(select_img, (int(x), int(y)), (int(x2), int(y2)), (0, 255, 0), 3)

    # Save detected result to detected_images directory
    combine_image = np.concatenate((tmp, select_img), axis=1)
    cv2.imwrite(detected_dir + "detected_" + b_img, combine_image)
    print("-------------------------------------------------------")

