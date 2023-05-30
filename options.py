import argparse
class TrainOptions():
  def __init__(self):
    self.parser = argparse.ArgumentParser()

    # data loader related
    self.parser.add_argument('--dataroot', type=str, default='./datasets/MSRS', help='path of data')
    self.parser.add_argument('--phase', type=str, default='train', help='phase for dataloading')
    self.parser.add_argument('--batch_size', type=int, default=4, help='batch size')
    self.parser.add_argument('--nThreads', type=int, default=16, help='# of threads for data loader')

    # training related
    self.parser.add_argument('--lr', default=1e-3, type=int, help='Initial learning rate for training model')
    self.parser.add_argument('--weight', default='dwa', type=str, help='multi-task weighting: equal, uncert, dwa')
    self.parser.add_argument('--n_ep', type=int, default=2500, help='number of epochs') # 400 * d_iter
    self.parser.add_argument('--n_ep_decay', type=int, default=1000, help='epoch start decay learning rate, set -1 if no decay') # 200 * d_iter
    self.parser.add_argument('--resume', type=str, default=None, help='specified the dir of saved models for resume the training')
    self.parser.add_argument('--gpu', type=int, default=0, help='GPU id')
    self.parser.add_argument('--temp', default=2.0, type=float, help='temperature for DWA (must be positive)')    
    
    # ouptput related
    self.parser.add_argument('--name', type=str, default='PSF', help='folder name to save outputs')
    self.parser.add_argument('--class_nb', type=int, default=9, help='class number for segmentation model')
    self.parser.add_argument('--display_dir', type=str, default='/data/timer/PSF/logs', help='path for saving display results')
    self.parser.add_argument('--result_dir', type=str, default='/data/timer/PSF/results', help='path for saving result images and models')
    self.parser.add_argument('--display_freq', type=int, default=10, help='freq (iteration) of display')
    self.parser.add_argument('--img_save_freq', type=int, default=10, help='freq (epoch) of saving images')
    self.parser.add_argument('--model_save_freq', type=int, default=10, help='freq (epoch) of saving models')
    
  def parse(self):
    self.opt = self.parser.parse_args()
    args = vars(self.opt)
    print('\n--- load options ---')
    for name, value in sorted(args.items()):
      print('%s: %s' % (str(name), str(value)))
    return self.opt

class TestOptions():
  def __init__(self):
    self.parser = argparse.ArgumentParser()

    # data loader related
    self.parser.add_argument('--dataroot', type=str, default='./datasets', help='path of data')
    self.parser.add_argument('--dataname', type=str, default='MSRS', help='name of dataset')
    self.parser.add_argument('--phase', type=str, default='test', help='phase for dataloading')
    self.parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    self.parser.add_argument('--nThreads', type=int, default=16, help='# of threads for data loader')    
    
    ## mode related
    self.parser.add_argument('--class_nb', type=int, default=9, help='class number for segmentation model')
    self.parser.add_argument('--resume', type=str, default='./results/PSF/checkpoints/best_model.pth', help='specified the dir of saved models for resume the training')
    self.parser.add_argument('--gpu', type=int, default=0, help='GPU id')
    
    # results related
    self.parser.add_argument('--name', type=str, default='PSFusion', help='folder name to save outputs')
    self.parser.add_argument('--result_dir', type=str, default='./Fusion_results', help='path for saving result images and models')
    
  def parse(self):
    self.opt = self.parser.parse_args()
    args = vars(self.opt)
    print('\n--- load options ---')
    for name, value in sorted(args.items()):
      print('%s: %s' % (str(name), str(value)))
    return self.opt
