import numpy as np
import cv2
import os

def mean_std(root='/data/timer/Segmentation/SegNeXt/datasets/MFNet/', data_type='PSFusion'):
    train_split = '/data/timer/Segmentation/SegNeXt/datasets/MFNet/split/train.txt'
    test_split = '/data/timer/Segmentation/SegNeXt/datasets/MFNet/split/test.txt'
    data_dir = os.path.join(root, data_type)    
    if os.path.isdir(data_dir): 
        means = [0, 0, 0]
        stdevs = [0, 0, 0]
        num_imgs = 0
        filelist = os.listdir(data_dir)
        split_list = []
        with open(train_split, 'r') as f:
            files = f.readlines()
            for item in files:
                file_name = item.strip()
                split_list.append(file_name + '.png')
                
        with open(test_split, 'r') as f:
            files = f.readlines()
            for item in files:
                file_name = item.strip()
                split_list.append(file_name + '.png')
        
        for item in filelist:
            if item in split_list:
                img_path = os.path.join(data_dir, item)
                num_imgs += 1
                img = cv2.imread(img_path)
                img = img.astype(np.float32) 
                for i in range(3):
                    means[i] += img[:, :, i].mean()
                    stdevs[i] += img[:, :, i].std()        
        means.reverse()
        stdevs.reverse()        
        means = np.asarray(means) / num_imgs
        stdevs = np.asarray(stdevs) / num_imgs
        means = np.around(means, 2)
        stdevs = np.around(stdevs, 2)
        
    return means.tolist(), stdevs.tolist()

if __name__ == '__main__':
    means, stdevs = mean_std()
    print('elif input_type == \'{}\': img_norm_cfg = dict(mean={}, std={}, to_rgb=True)'.format('PSFusion', means, stdevs))