import matplotlib.pyplot as plt
from tqdm import trange

from utils import *
from dataloader import *
import pdb
from model import HS_G
from args import get_args
from torchmetrics.image import StructuralSimilarityIndexMeasure


def init_training(args):
    # load psf
    psf = np.load('./data/PSF/sim/psf.npy') # mode='sim' or 'real'. Note that the loaded psf should be already normalized

    psf = torch.from_numpy(psf).float().to(args.device)
    psf = torch.flip(psf, dims=(-1,-2))

    # load dataset
    transform = transforms.Compose([
        transforms.RandomCrop([args.img_size,args.img_size],pad_if_needed=True),
        transforms.RandomHorizontalFlip(),
        transforms.Pad(args.scene_padding, padding_mode='reflect')
        ])

    transform_val = transforms.Compose([
        transforms.CenterCrop([args.img_size,args.img_size]),
        transforms.Pad(args.scene_padding, padding_mode='reflect')
        ]) 

    trainset1 = ICVL(transform = transform)
    trainset2 = Harvard(transform = transform)
    trainset = torch.utils.data.ConcatDataset([trainset1, trainset2])
    
    valset = KAUST(transform=transform_val)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size = args.batch_size, shuffle=True)
    valloader = torch.utils.data.DataLoader(valset, batch_size = args.batch_size, shuffle=False)
    
    # set model, optimizer, and loss
    HS_model = HS_G(input_ref = 2, input_enc = 2, RGB=args.use_RGB).to(args.device)

    HS_optimizer = torch.optim.Adam(params=HS_model.parameters(), lr = args.lr)
    HS_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(HS_optimizer, 'min', factor=0.9, patience=2, min_lr=0.1*args.lr)

    # if we need to use measurements in the loss function, pad the psf to the shape of padded scene. Otherwise, just pad the psf to the original shape of the scene.
    args.psf_padding = (args.img_size - args.psf_size)//2

    if args.use_RGB:
        camera_response = torch.load('./data/basler_gc_response.pt').to(args.device)
        camera_response = camera_response[[2, 1, 0], 5:, ...]
        args.qe = camera_response
    
    loss_fn = Loss(args)

    return psf, trainloader, valloader, HS_model, HS_optimizer, HS_scheduler, loss_fn


def train(args):
    set_seed(622, deterministic=True)
    psf, trainloader, valloader, HS_model, HS_optimizer, HS_scheduler, loss_fn = init_training(args)
    
    # pdb.set_trace()

    start_epoch = 0 
    train_loss = []

    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(args.device)
    sam = SpectralAngleMapper().to(args.device)

    # HS_model.load_state_dict(torch.load('./HS_weight.pt', weights_only=True))

    for epoch in trange(start_epoch, start_epoch+args.num_epoch):
        HS_model.train()
        epoch_loss = 0.0
        train_PSNR = 0.0
        train_SSIM = 0.0
        train_SAM  = 0.0

        for _, batch_data in enumerate(trainloader):
            HS_optimizer.zero_grad()
            
            batch_gt = batch_data['scene']

            batch_meas = image_formation(batch_gt, psf, args)
            
            output_hs = HS_model(batch_meas, torch.nn.functional.pad(psf,(args.psf_padding, args.psf_padding+1, args.psf_padding, args.psf_padding+1)))# the DWDN model requires the psf to have the same shape as input
            batch_gt = batch_gt[:,:,args.scene_padding:-args.scene_padding,args.scene_padding:-args.scene_padding]
            loss = loss_fn(output_hs, batch_gt)

            loss.backward()

            HS_optimizer.step()

            epoch_loss += loss.item()

            with torch.no_grad():
                train_PSNR += torch.sum(batch_psnr(output_hs,batch_gt))
                train_SSIM += torch.sum(ssim(output_hs, batch_gt))
                train_SAM  += torch.sum(sam(output_hs, batch_gt))
                

        train_loss.append(epoch_loss/len(trainloader.dataset))

        if epoch >= args.start_decay_epoch:
            HS_scheduler.step(epoch_loss/len(trainloader.dataset))


        plt.figure()
        plt.plot(np.arange(epoch+1), np.array(train_loss))
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.ylim(bottom=0)
        plt.grid(True)
        plt.savefig('./learning_curve.png')
        plt.close()


        print(f"\n Epoch [{epoch+1}/{args.num_epoch}], Loss: {epoch_loss/len(trainloader.dataset)}, PSNR: {train_PSNR/len(trainloader.dataset)}, SSIM: {train_SSIM/len(trainloader.dataset)}, SAM:{train_SAM/len(trainloader.dataset)}")
    
        if (epoch+1) % 10 == 0:
            HS_model.eval()
            val_loss = 0.0
            val_PSNR = 0.0
            val_SSIM = 0.0
            val_SAM  = 0.0

            with torch.no_grad():
                for _, batch_data in enumerate(valloader):
                    batch_gt = batch_data['scene']
                    batch_meas = image_formation(batch_gt, psf, args)

                    output_hs = HS_model(batch_meas, torch.nn.functional.pad(psf,(args.psf_padding, args.psf_padding+1, args.psf_padding, args.psf_padding+1)))# the DWDN model requires the psf to have the same shape as input
                    
                    batch_gt = batch_gt[:,:,args.scene_padding:-args.scene_padding,args.scene_padding:-args.scene_padding]

                    loss = loss_fn(output_hs, batch_gt)
                
                    # loss = perceptual_loss(torch.sum(outputs,dim=1,keepdim=True), torch.sum(batch_gt,dim=1,keepdim=True))
                    val_loss += loss.item()
                    val_PSNR += torch.sum(batch_psnr(output_hs,batch_gt))
                    val_SSIM += torch.sum(ssim(output_hs, batch_gt))
                    val_SAM  += torch.sum(sam(output_hs, batch_gt))

            # torch.save(HS_model.state_dict(), f'./_step{epoch+1}.pt')
            print(f"Validation set: Loss: {val_loss/len(valloader.dataset)}, PSNR:{val_PSNR/len(valloader.dataset)}, SSIM: {val_SSIM/len(valloader.dataset)}, SAM:{val_SAM/len(valloader.dataset)}")

    pdb.set_trace()

if __name__ == "__main__":
    args = get_args()
    train(args)