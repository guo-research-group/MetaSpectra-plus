import numpy as np
import random
import os
from torchvision import transforms
import torch
from torchmetrics.image import SpectralAngleMapper
import pdb
import lpips

class Loss(torch.nn.Module):
    def __init__(self, args):
        super().__init__()

        self.px_loss = torch.nn.L1Loss().to(args.device) if args.loss_fn['l1'] else torch.nn.MSELoss().to(args.device)
        self.spec_loss = SpectralAngleMapper().to(args.device) if args.loss_fn['SAM'] else None
        self.percp_loss = lpips.LPIPS(net='vgg').to(args.device)
        self.QE_weight = args.qe

        self.w_px = args.loss_fn['w_px']
        self.w_sam = args.loss_fn['w_sam']

    def forward(self, output, gt):
        
        # Compare the RGB image using the perceptual loss + SAM loss
        out_spectral = torch.nn.functional.conv2d(input=output, weight=self.QE_weight)
        gt_spectral = torch.nn.functional.conv2d(input=gt, weight=self.QE_weight)
        
        if self.spec_loss is not None:
            # pdb.set_trace()
            return self.w_px * self.percp_loss(2*out_spectral-1, 2*gt_spectral-1).mean() + self.w_sam * self.spec_loss(output, gt) #self.w_px * self.px_loss(output, gt) + self.w_sam * self.spec_loss(output, gt)
        else:
            return self.w_px * self.px_loss(output, gt)

 
def set_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if deterministic:
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.backends.cudnn.enabled = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def batch_psnr(recon, gt):
    mse = torch.mean((recon - gt) ** 2, dim=(-1, -2, -3))
    return 20 * torch.log10(1 / torch.sqrt(mse))



def image_formation(scene, psf_list, args):
    '''
    Input:
        scene: (batch, channel, height, weight)
        psf_list: (# psf, channel, height, weight)

    Output:
        measurement: (# psf, batch, channel (1 or 3), height, weight)
    '''
    if not isinstance(psf_list,torch.Tensor):
        psf_list = torch.from_numpy(psf_list).float().to(args.device)

    num_meas = psf_list.shape[0]
    psf_list = psf_list.permute((1,0,2,3)).flatten(0,1).unsqueeze(1) # shape (num, channel, h, w) -> (num*channel, 1, h, w) 
    measurement = torch.nn.functional.conv2d(input=scene, weight=psf_list, padding=args.conv_pad_mode, groups=scene.shape[1]) # shape (batch, num*channel, h, w)

    noise_level = args.noise_min + (args.noise_max - args.noise_min) * torch.rand(1)

    if args.use_RGB:
        measurement = torch.stack([torch.nn.functional.conv2d(input=measurement[:,i::num_meas,:,:], weight=args.qe) for i in range(num_meas)])
    else:
        measurement = torch.stack([torch.nn.functional.conv2d(input=measurement[:,i::num_meas,:,:], weight=args.qe) for i in range(num_meas)]) 
        

    measurement += torch.randn_like(measurement)*noise_level.to(args.device)
    
    return measurement.clamp(0,1)
