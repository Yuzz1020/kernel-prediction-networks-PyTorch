import cv2
import numpy as np
import os

# reserved for future use
# #read images into greayscale and add poisson noise
# def read_images(image_path, image_count):
#     for i in range(image_count):
#         if i % 50 == 0:
#             if i != 0:
#                 #concatenate images
#                 images_np = np.concatenate(images, axis=2)
#                 print(images_np.shape)
#                 #save to npy file
#                 np.save(image_path + "images_" + str(i//50-1) + ".npy", images_np)
#             images = []
#         tmp = cv2.imread(image_path + "frame%d.jpg" % i, 0)
#         #magnitude normalization (to 0.5)
#         tmp = tmp / np.max(tmp) * 0.5
#         #add poisson noise
#         tmp = np.random.poisson(tmp, tmp.shape)
#         tmp = (tmp > 0).astype(np.float32)
#         # #save tmp as image
#         # cv2.imwrite("test.jpg", tmp.astype(np.uint8)*255)
#         # exit()
#         images.append(np.expand_dims(tmp,2))


#convert mp4 to list of images
def convert_video_to_images(video_path, image_path):
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    print(success)
    count = 0
    while success:
        cv2.imwrite(image_path + "frame%d.jpg" % count, image)  # save frame as JPEG file
        success, image = vidcap.read()
        print('Read a new frame: ', success)
        count += 1
    return count

#read images into greayscale and add poisson noise
def read_images(image_path, train_path, label_path, image_count, oversampled_rate=50):
    if not os.path.exists(train_path):
            os.makedirs(train_path)
    if not os.path.exists(label_path):
            os.makedirs(label_path)
    for i in range(image_count):
        tmp = cv2.imread(image_path + "frame%d.jpg" % i, 0)
        np.save(label_path + "frame%d.npy" % i, tmp)
        #magnitude normalization (to 0.5)
        tmp = tmp / np.max(tmp) * 0.5
        images = []
        for j in range(oversampled_rate):
            #add poisson noise
            tmp_poisson = np.random.poisson(tmp, tmp.shape)
            tmp_poisson = (tmp_poisson > 0).astype(np.float32)
            # #save tmp as image
            # cv2.imwrite("test.jpg", tmp_poisson.astype(np.uint8)*255)
            # exit()
            images.append(np.expand_dims(tmp_poisson,2))
        #concatenate images
        images_np = np.concatenate(images, axis=2)
        print("frame%d.npy" % i, images_np.shape)
        #save to npy file
        np.save(train_path + "frame%d.npy" % i, images_np)

#convert_video_to_images("../original_high_fps_videos/GOPR9654a.mp4","../test_images/")
read_images("../test_images/","../train/","../label/", 1400)
