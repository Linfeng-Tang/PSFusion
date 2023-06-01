 #!/usr/bin/python
# -*- encoding: utf-8 -*-


from logger import setup_logger
from model_TII import BiSeNet
from MSRS import MSRS
from loss import OhemCELoss,NormalLoss
from fullevaluate import evaluate
from optimizer import Optimizer

import torch
from torch.utils.data import DataLoader
import os
import os.path as osp
import logging
import time
import datetime
import argparse
import torch.nn as nn

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def parse_args():
    parse = argparse.ArgumentParser()
    return parse.parse_args()

def train(respth='./res_MFNet', datapth='./MFNet', Method='NestFuse', data_Method='NestFuse', prepth=None):
    """
    respth = './res_MFNet' 模型存放的上级目录，同时也是log日志存放的目录
    datapth='./MFNet'      数据存放的文件夹 images位于 datapth/'dataMethod' labels位于： datapth/Label
    Method = 'NestFuse'    模型保存是的子文件夹，用于区分不同融合方法生成的融合结果得到分割模型
    data_Method = 'NestFuse' 用于区分不同融合算法生成的融合图像，由于有时候要在同一个数据集上训练多个模型所以使用Method 和dataMethod区分
    prepth = None          如果需要加载预训练的模型则使用prepth指定预训练模型的路径    
    """
    respth = os.path.join(respth, Method)
    if not osp.exists(respth):
        os.makedirs(respth)
    logger = logging.getLogger()
    setup_logger(respth)
    ## dataset
    n_classes = 9
    n_img_per_gpu = 16 # batch_size
    n_workers = 4
    cropsize = [640, 480]    
    ds = MSRS(datapth, cropsize=cropsize, mode='train', Method=data_Method)
    dl = DataLoader(
        ds,
        batch_size=n_img_per_gpu,
        shuffle=False,
        num_workers=n_workers,
        pin_memory=True,
        drop_last=True,
    )

    ## model
    ignore_idx = 255
    net = BiSeNet(n_classes=n_classes)
    if not prepth == None:
        net.load_state_dict(torch.load(prepth))
        best_mIou = evaluate(respth=respth, dspth=datapth, Method=data_Method, save_pth=prepth)
    else:
        best_mIou = 0.0
    net.cuda()
    net.train()
    score_thres = 0.7
    n_min = n_img_per_gpu * cropsize[0] * cropsize[1] // 16    
    criteria_p = OhemCELoss(thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx).cuda()
    criteria_16 = OhemCELoss(thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx).cuda()
    
    ## optimizer
    momentum = 0.9
    weight_decay = 5e-4
    lr_start = 1e-3
    max_iter = 160000
    power = 0.9
    warmup_steps = 1000
    warmup_start_lr = 1e-5
    optim = Optimizer(
        model=net,
        lr0=lr_start,
        momentum=momentum,
        wd=weight_decay,
        warmup_steps=warmup_steps,
        warmup_start_lr=warmup_start_lr,
        max_iter=max_iter,
        power=power,
    )

    ## train loop
    msg_iter = 100
    loss_avg = []
    st = glob_st = time.time()
    diter = iter(dl)
    epoch = 0
    for it in range(max_iter):
        try:
            im, lb, _ = next(diter)
            if not im.size()[0] == n_img_per_gpu:
                raise StopIteration
        except StopIteration:
            epoch += 1
            # sampler.set_epoch(epoch)
            diter = iter(dl)
            im, lb, _ = next(diter)
        im = im.cuda()
        lb = lb.cuda()
        lb = torch.squeeze(lb, 1)

        optim.zero_grad()
        out, mid = net(im)
        lossp = criteria_p(out, lb)
        loss2 = criteria_16(mid, lb)
        loss = 10 * lossp + 7.5 * loss2
        loss.backward()
        optim.step()

        loss_avg.append(loss.item())
        ## print training log message
        if (it + 1) % msg_iter == 0:
            loss_avg = sum(loss_avg) / len(loss_avg)
            lr = optim.lr
            ed = time.time()
            t_intv, glob_t_intv = ed - st, ed - glob_st
            eta = int((max_iter - it) * (glob_t_intv / it))
            eta = str(datetime.timedelta(seconds=eta))
            msg = ', '.join(
                [
                    'it: {it}/{max_it}',
                    'lr: {lr:4f}',
                    'loss: {loss:.4f}',
                    'eta: {eta}',
                    'time: {time:.4f}',
                ]
            ).format(
                it=it + 1, max_it=max_iter, lr=lr, loss=loss_avg, time=t_intv, eta=eta
            )
            logger.info(msg)
            loss_avg = []
            st = ed
        if (it + 1) % 1000 == 0 and (it + 1 ) > 50000: #
            ## dump the final model
            save_pth = osp.join(respth, 'model_{}.pth'.format('temp'))
            net.cpu()
            state = net.module.state_dict() if hasattr(net, 'module') else net.state_dict() 
            torch.save(state, save_pth)
            test_mIou = evaluate(respth=respth, dspth=datapth, Method=data_Method, save_pth=save_pth)            
            logger.info('The best mIoU is {:.4f}, the test mIoU is: {:.4f}'.format(best_mIou, test_mIou))
            if test_mIou > best_mIou:
                best_mIou = test_mIou
                save_pth = osp.join(respth, 'model_final.pth')
                torch.save(state, save_pth)                     
                logger.info('training done, the best mIou is: {:.4f},  the best model is saved to: {}'.format(best_mIou, save_pth))    
            net.cuda()
    ## dump the final model
    # save_pth = osp.join(respth, 'model_final.pth')
    # net.cpu()
    # state = net.module.state_dict() if hasattr(net, 'module') else net.state_dict()
    # torch.save(state, save_pth)
    # logger.info('training done, model saved to: {}'.format(save_pth))


if __name__ == "__main__":
    respth='./res_MSRS'
    datapth='./dataset/MSRS'
    Method='PSFusion'
    data_Method='PSFusion'
    prepth= None
    train(respth=respth, datapth=datapth, Method=Method, data_Method=data_Method, prepth=prepth)
#  evaluate()
