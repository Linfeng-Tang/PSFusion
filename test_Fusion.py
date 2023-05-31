from create_dataset import *
from utils import *
from PSF import PSF
from options import * 
from saver import resume, save_img_single
from tqdm import tqdm
from thop import profile

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    # parse options    
    parser = TestOptions()
    opts = parser.parse()
    # define model, optimiser and scheduler
    device = torch.device("cuda:{}".format(opts.gpu) if torch.cuda.is_available() else "cpu")
    MPF_model = PSF(opts.class_nb).to(device)
    MPF_model = resume(MPF_model, model_save_path=opts.resume, device=device, is_train=False)
    
    # define dataset    
    test_dataset = FusionData(opts)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=opts.batch_size,
        shuffle=False)
    
    # Train and evaluate multi-task network
    multi_task_tester(test_loader, MPF_model, device, opts)
    
def multi_task_tester(test_loader, multi_task_model, device, opts):
    multi_task_model.eval()
    is_rgb = False ## 用来标记重建的可见光图像是彩色图像还是灰度图像。
    test_bar= tqdm(test_loader)
    seg_metric = SegmentationMetric(opts.class_nb, device=device)
    lb_ignore = [255]
    ## define save dir
    Fusion_save_dir = os.path.join('./Fusion_results', opts.dataname, opts.name)
    # Fusion_save_dir = os.path.join('./Biseg/PSFusion_5180/', 'train')
    # Fusion_save_dir = os.path.join('/data/timer/Segmentation/SegNext/datasets/MSRS/PSFusion')
    os.makedirs(Fusion_save_dir, exist_ok=True)
    # Seg_save_dir = os.path.join(save_root, 'Segmentation', opts.dataname)
    # os.makedirs(Seg_save_dir, exist_ok=True)
    with torch.no_grad():  # operations inside don't track history
        for it, (img_ir, img_vi, img_names, widths, heights) in enumerate(test_bar):
            img_ir = img_ir.to(device)
            img_vi = img_vi.to(device)
            vi_Y, vi_Cb, vi_Cr = RGB2YCrCb(img_vi)
            vi_Y = vi_Y.to(device)
            vi_Cb = vi_Cb.to(device)
            vi_Cr = vi_Cr.to(device)       
            if it == 0:
                flops, params = profile(multi_task_model,inputs=(img_vi, img_ir))
                print('flops: {:.2f} G | params: {:.2f} M'.format(flops / (1024* 1024 * 1024), params / (1024* 1024)))
            Seg_pred, _, _, fused_img, _, _  = multi_task_model(img_vi, img_ir)  
            # seg_result = torch.argmax(Seg_pred, dim=1, keepdim=True)
            fused_img = YCbCr2RGB(fused_img, vi_Cb, vi_Cr)
            for i in range(len(img_names)):
                img_name = img_names[i]
                # seg_save_name = os.path.join(Seg_save_dir, img_name)
                fusion_save_name = os.path.join(Fusion_save_dir, img_name)
                # seg_visualize(seg_result[i, ::].unsqueeze(0).squeeze(dim=1), seg_save_name)
                save_img_single(fused_img[i, ::], fusion_save_name, widths[i], heights[i])
                test_bar.set_description('Image: {} '.format(img_name))
    
if __name__ == '__main__':
    main()
