import argparse

def get_args():
    parser = argparse.ArgumentParser()

    # HS reconstruction setting
    parser.add_argument('--device', type = str, default = 'cuda', help = 'device')
    parser.add_argument('--psf_size', type = int, default = 61, help = 'pixel number of psf')
    parser.add_argument('--img_size', type = int, default = 512, help = 'pixel number of ouput image')
    parser.add_argument('--conv_pad_mode', type = str, default = 'valid', help = 'padding method of conv2d')
    parser.add_argument('--scene_padding', type = int, default = 30, help = '# 0s to be padded to the ground truth HS cube') # 64 is a number that suits DWDN, it can be 25 if we don't use DWDN
    parser.add_argument('--psf_padding', type = int, default = 0, help = '# 0s to be padded to the PSF, this is necessary for fft in DWDN')
    parser.add_argument('--noise_min', type = float, default = 0.001, help = 'minimum standard deviation of Gaussian noise')
    parser.add_argument('--noise_max', type = float, default = 0.01, help = 'maximum standard deviation of Gaussian noise')
    parser.add_argument('--use_RGB', type = bool, default = True, help = 'use RGB image or grayscale image')
   
    parser.add_argument('--loss_fn', type = dict, default = {'l1': False, 'l2': False, 'SAM': True, 'w_px': 1, 'w_sam': 1}, help = 'loss functions and weights')
    parser.add_argument('--lr', type = float, default = 1e-4, help = 'learning rate of HS reconstruction model')
    parser.add_argument('--batch_size', type = int, default = 1, help = 'batch size')
    parser.add_argument('--num_epoch', type = int, default = 100, help = 'number of epochs')
    parser.add_argument('--start_decay_epoch', type = int, default = 5, help = 'epoch that starts a cosine learning rate')

    return parser.parse_args()