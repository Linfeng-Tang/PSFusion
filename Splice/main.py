import os
from natsort import natsorted
from tqdm import tqdm
import matplotlib.pyplot as plt
import string
from matplotlib.font_manager import FontProperties
import numpy as np
if __name__ == '__main__':
    font = {'family': 'Times New Roman',
            'weight': 'normal',
            'size': 14,
            }
    plt.rc('font', **font)
    dataroot = r'/data/timer/Segmentation/SegNeXt/datasets'
    results_root = '/data/timer/Segmentation/SegNeXt/datasets'
    dataset_list = ['MSRS']
    method_list = ['Infrared', 'Visible', 'GTF', 'DIDFuse', 'RFN-Nest', 'FusionGAN', 'TarDAL', 'UMF-CMGR', 'SeAFusion', 'SwinFusion', 'U2Fusion', 'PSF']
    for dataset in dataset_list:
        ir_dir = os.path.join(dataroot, dataset, 'Thermal')
        vi_dir = os.path.join(dataroot, dataset, 'RGB')
        f_dir = os.path.join(results_root, dataset)
        save_dir = os.path.join(results_root, dataset, 'Splice')
        os.makedirs(save_dir, exist_ok=True)
        filelist = natsorted(os.listdir(ir_dir))
        test_bar = tqdm(filelist)
        # index = filelist.index('02821.png')
        for num, item in enumerate(test_bar):
            # if num < index:
            #     continue
            fig, axes = plt.subplots(nrows=2, ncols=6, figsize=(12, 4))
            # 关闭所有子图的坐标轴和标尺
            for ax in axes.ravel():
                ax.set_axis_off()
                ax.tick_params(labelbottom=False, labelleft=False)
            save_path = os.path.join(save_dir, item)
            for i, (method, ax) in enumerate(zip(method_list, axes.ravel())):
                if method == 'Infrared':
                    img_path = os.path.join(ir_dir, item)
                elif method == 'Visible':
                    img_path = os.path.join(vi_dir, item)
                else:
                    img_path = os.path.join(f_dir, method, item)
                img = plt.imread(img_path)
                gray_flag = False
                if img.ndim == 2:
                    gray_flag = True
                else:
                    gray_flag = False
                # 显示图片
                if gray_flag:
                    ax.imshow(img, cmap='gray')
                else:
                    ax.imshow(img)
                # 显示图片名称
                # font = FontProperties(fname='Times New Roman.ttf')
                number = '({})'.format(string.ascii_lowercase[i])
                ax.set_title('{} {}'.format(number, method), fontsize=10, y=-0.2)            # 调整子图布局

            plt.subplots_adjust(left=0, right=0.9, bottom=0, top=0.9, wspace=0.01, hspace=-0.15)
            # plt.tight_layout()
            # 显示图像
            # plt.show()
            plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches = 0 )
            plt.close()
            test_bar.set_description('{} | {}'.format(dataset, item))