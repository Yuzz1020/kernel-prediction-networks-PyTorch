import cv2
import numpy as np
import os

# reserved for future use: no oversampling; directly add the poisson noise and concatenate images
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

fixed_idx = False
if fixed_idx:
    idx_w = 200
    idx_h = 100

def crop_random(img, scale_factor, w, h=None):
    """randomly crop a patch shaped patch_size*patch_size, with a upscale factor"""
    h = w if h is None else h
    nw = img.shape[1] - w*scale_factor
    nh = img.shape[0] - h*scale_factor
    if nw < 0 or nh < 0:
        raise RuntimeError("Image is to small {} for the desired size {}". \
                                format((img.shape[1], img.shape[0]), (w*scale_factor, h*scale_factor))
                          )
    
    if not fixed_idx:
        idx_w = np.random.randint(0, nw+1)
        idx_h = np.random.randint(0, nh+1)

    scaled_patch = img[idx_h:idx_h+h*scale_factor, idx_w:idx_w+w*scale_factor]
    # print(scaled_patch.shape)

    patch = cv2.resize(scaled_patch, (w, h), interpolation=cv2.INTER_CUBIC)
    # print(patch.shape)
    return patch




#convert mp4 to list of images
def convert_video_to_images(video_paths, image_path):
    count = 0
    for video_path in video_paths:
        vidcap = cv2.VideoCapture(video_path)
        success, image = vidcap.read()
        print(success)
        while success:
            #! moved the crop to the dataset loader
            # image = crop_random(image, 2, 640, 640)
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
        if np.max(tmp) == 0:
            tmp = tmp * 0.0
        else:
            tmp = tmp / np.max(tmp) * 0.5
        images = []
        for j in range(oversampled_rate):
            #add poisson noise
            tmp_poisson = np.random.poisson(tmp, tmp.shape)
            tmp_poisson = (tmp_poisson > 0).astype(np.float32)
            # if i ==144:
            #     #save tmp as image
            #     cv2.imwrite(train_path+"/test_"+str(i)+"_"+str(j)+".jpg", tmp_poisson.astype(np.uint8)*255)
            
            #concatenate the channels: the last dimension
            images.append(np.expand_dims(tmp_poisson,2))
        #concatenate images
        images_np = np.concatenate(images, axis=2)
        print("frame%d.npy" % i, images_np.shape)
        #save to npy file
        np.save(train_path + "frame%d.npy" % i, images_np)

video_list=[]
for f in os.listdir("../original_high_fps_videos/"):
    print(f)
    if "GOPR9646.mp4" not in f:
        video_list.append("../original_high_fps_videos/"+f)

#training images
img_count = convert_video_to_images(video_list, "../test_images/")
print("total images: ", img_count)

# no longer needed; merged to the dataloader
# read_images("../test_images/","../train/","../label/", 121340)

#evaluation images
img_count = convert_video_to_images(["../original_high_fps_videos/GOPR9646.mp4"], "../eval_images/")
print("total images: ", img_count)

# no longer needed; merged to the dataloader
# read_images("../eval_images/","../eval/","../eval_label/", 671)
