from create_dataset import *
from utils import *
from PSF import PSF
from options import * 
from saver import resume, save_img_single
from tqdm import tqdm

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num / 1024 / 1024, 'Trainable': trainable_num}


def main():
    # parse options    
    parser = TestOptions()
    opts = parser.parse()
    # define model, optimiser and scheduler
    device = torch.device("cuda:{}".format(opts.gpu) if torch.cuda.is_available() else "cpu")
    MPF_model = PSF(opts.class_nb).to(device)
    MPF_model = resume(MPF_model, model_save_path=opts.resume, device=device, is_train=False)
    
    # define dataset    
    test_dataset = MSRSData(opts, is_train=False)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=opts.batch_size,
        shuffle=False)
    
    # Train and evaluate multi-task network
    multi_task_tester(test_loader, MPF_model, device, opts)
    
def multi_task_tester(test_loader, multi_task_model, device, opts):
    print(get_parameter_number(multi_task_model))
    multi_task_model.eval()
    is_rgb = False ## 用来标记重建的可见光图像是彩色图像还是灰度图像。
    test_bar= tqdm(test_loader)
    seg_metric = SegmentationMetric(opts.class_nb, device=device)
    lb_ignore = [255]
    ## define save dir
    save_root = os.path.join(opts.result_dir, opts.name)
    Fusion_save_dir = os.path.join(save_root, 'MPF', 'tarin', 'MSRS')
    
    # Fusion_save_dir = '/data/timer/Segmentation/SegFormer/datasets/MSRS/MPF'
    os.makedirs(Fusion_save_dir, exist_ok=True)
    Seg_save_dir = os.path.join(save_root, 'Segmentation')
    os.makedirs(Seg_save_dir, exist_ok=True)
    Re_vis_save_dir = os.path.join(save_root, 'Reconstruction_Vis')
    os.makedirs(Re_vis_save_dir, exist_ok=True)
    Re_ir_save_dir = os.path.join(save_root, 'Reconstruction_IR')
    os.makedirs(Re_ir_save_dir, exist_ok=True)
    with torch.no_grad():  # operations inside don't track history
        for it, (img_ir, img_vi, label, img_names) in enumerate(test_bar):
            img_ir = img_ir.to(device)
            img_vi = img_vi.to(device)
            vi_Y, vi_Cb, vi_Cr = RGB2YCrCb(img_vi)
            vi_Y = vi_Y.to(device)
            vi_Cb = vi_Cb.to(device)
            vi_Cr = vi_Cr.to(device)
            label = label.to(device)           
            Seg_pred, _, _, fused_img, re_vi, re_ir  = multi_task_model(img_vi, img_ir)    
            # re_vi = torch.clamp(re_vi, 0, 1)
            # re_ir = torch.clamp(re_ir, 0, 1)
            # fused_img = torch.clamp(fused_img, 0, 1)
            # print(torch.min(fused_img), torch.max(fused_img))        
            seg_result = torch.argmax(Seg_pred, dim=1, keepdim=True) ## print(seg_result.shape())
            seg_metric.addBatch(seg_result, label, lb_ignore)
            # conf_mat.update(seg_result.flatten(), label.flatten())
            # compute mIoU and acc
            fused_img = YCbCr2RGB(fused_img, vi_Cb, vi_Cr)
            # if not is_rgb:
            #     re_vi = YCbCr2RGB(re_vi, vi_Cb, vi_Cr)
            for i in range(len(img_names)):
                img_name = img_names[i]
                seg_save_name = os.path.join(Seg_save_dir, img_name)
                fusion_save_name = os.path.join(Fusion_save_dir, img_name)
                vi_save_name = os.path.join(Re_vis_save_dir, img_name)
                ir_save_name = os.path.join(Re_ir_save_dir, img_name)
                # seg_visualize(seg_result[i, ::].unsqueeze(0).squeeze(dim=1), seg_save_name)
                # save_img_single(fused_img[i, ::], fusion_save_name)
                # save_img_single(re_vi[i, ::], vi_save_name)
                # save_img_single(re_ir[i, ::], ir_save_name)
                test_bar.set_description('Image: {} '.format(img_name))
        IoU =seg_metric.IntersectionOverUnion()
        IoU = [np.array(a.item()) for a in IoU]
        mIoU = np.array(seg_metric.meanIntersectionOverUnion().item())
        Acc = np.array(seg_metric.pixelAccuracy().item())
        IoU_list = IoU
        IoU_list.append(mIoU)
        IoU_list = [np.round(100 * i, 2) for i in IoU_list]
        print('IoU:', IoU_list, 'ACC: {:.4f}'.format(Acc))

    
if __name__ == '__main__':
    main()
