import torch

from model import *
from urban_dataset import *

from torch.utils.tensorboard import SummaryWriter
from torch.optim import lr_scheduler
import shutil
import platform

K = 32768
epoch = 500
dim = 256
batch_size = 512
t = 0.07
m = 0.999
lr = 5e-4
device = 'cuda:0'
system_type = platform.system()
# 'Windows' 'Linux'
proj_path = None
data_path = None
train_txt = None
test_text = None
if system_type == 'Windows':
    proj_path = 'E:/00_Code/PyCharmProjects/UrbanSceneNet/'
    data_path = 'F:/UrbanSceneImagePatches/'
    train_txt = 'E:/00_Code/PyCharmProjects/UrbanSceneNet/DataProcess/txt/UrbanScenePatch/train_image_win.txt'
    test_txt = 'E:/00_Code/PyCharmProjects/UrbanSceneNet/DataProcess/txt/UrbanScenePatch/test_image_win.txt'
if system_type == 'Linux':
    proj_path = '/home/ubuntu/workdic/UrbanSceneNet/urban-scene-seg-net/'
    data_path = '/home/ubuntu/workdic/UrbanSceneImagePatches/'
    train_txt = '/home/ubuntu/workdic/UrbanSceneNet/urban-scene-seg-net/DataProcess/txt/UrbanScenePatch/train_image_linux.txt'
    test_txt = '/home/ubuntu/workdic/UrbanSceneNet/urban-scene-seg-net/DataProcess/txt/UrbanScenePatch/test_image_linux.txt'

train_dataset = GlassDiscDataSet(txt_file=train_txt)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


def train_one_epoch(_, model, optimizer, criterion, data_loader, device, all_epoch, lr, writer=None):
    model.train()
    loss = None
    for batch_idx, (img_q, img_k) in enumerate(data_loader):
        img_q = img_q.to(device)
        img_k = img_k.to(device)
        outputs, labels = model(img_q, img_k)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if writer is not None:
            writer.add_scalar(tag='loss_for_epoch{}/train'.format(_ + 1),
                              scalar_value=loss.item(),
                              global_step=batch_idx)
        print('Train Epoch: {}/{}--[{}/{} ({:.0f}%)]--****--Loss: {:.6f}--Lr:{:.6f}'.format(_ + 1, all_epoch,
                                                                                            batch_idx * len(img_q),
                                                                                            len(data_loader.dataset),
                                                                                            100. * batch_idx / len(
                                                                                                data_loader),
                                                                                            loss.item(),
                                                                                            lr))
    return loss.item()


def save_checkpoint(state, filename, is_best=False):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, "model_best.pth")


def load_params(model, optimizer, checkpoint, scheduler=None):
    model_weight = torch.load(checkpoint)["state_dict"]
    model.load_state_dict(model_weight)
    optimizer.load_state_dict(torch.load(checkpoint)["optimizer"])
    epoch = torch.load(checkpoint)["epoch"]
    loss_now = torch.load(checkpoint)["loss"]
    return epoch, loss_now


def train_single_gpu(checkpoint=None):
    model = MoCo(dim=dim, K=K, T=t, m=m).to(device)
    loss_fn = nn.CrossEntropyLoss().to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = lr_scheduler.CosineAnnealingLR(optim, 20, eta_min=1e-6)
    loss_temp = 10.0
    start_epoch = 0
    if checkpoint is not None:
        start_epoch, loss_temp = load_params(model=model,
                                             optimizer=optim,
                                             checkpoint=checkpoint)
    writer = SummaryWriter(log_dir=proj_path + 'Segmentation/UnsupervisedGlassDetection/log/', flush_secs=120)
    for _ in range(epoch - start_epoch):
        writer.add_scalar(tag='lr_epoch/train',
                          scalar_value=optim.param_groups[0]['lr'],
                          global_step=_ + start_epoch + 1)
        loss_now = train_one_epoch(_=_ + start_epoch, model=model, optimizer=optim, criterion=loss_fn,
                                   data_loader=train_loader, device=device, all_epoch=epoch, writer=writer,
                                   lr=optim.param_groups[0]['lr'])
        writer.add_scalar(tag='loss_epoch/train',
                          scalar_value=loss_now,
                          global_step=_ + start_epoch + 1)
        if (_ + 1) >= 10:
            if (loss_now < loss_temp) or (_ + start_epoch % 5 == 0):
                is_best = False
                if loss_now < loss_temp:
                    is_best = True
                loss_temp = loss_now
                save_checkpoint(
                    {
                        "initial_lr": lr,
                        "epoch": _ + start_epoch + 1,
                        "loss": loss_now,
                        "state_dict": model.state_dict(),
                        "optimizer": optim.state_dict(),
                        "scheduler": scheduler.state_dict(),
                    },
                    is_best=is_best,
                    filename=proj_path + "Segmentation/UnsupervisedGlassDetection/checkpoints/checkpoint_{:04d}.pth".format(
                        start_epoch + _ + 1),
                )
        scheduler.step()
    writer.close()


def train_multi_gpu(main_device='cuda:0', device_ids=[0, 1, 2, 3], checkpoint=None):
    model = MoCo(dim=dim, K=K, T=t, m=m).to(main_device)
    loss_fn = nn.CrossEntropyLoss().to(main_device)
    model = nn.DataParallel(model, device_ids=device_ids)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = lr_scheduler.CosineAnnealingLR(optim, 20, eta_min=1e-6)
    loss_temp = 10.0
    start_epoch = 0
    if checkpoint is not None:
        start_epoch, loss_temp = load_params(model=model,
                                             optimizer=optim,
                                             checkpoint=checkpoint,
                                             parallel=True)
    writer = SummaryWriter(log_dir=proj_path + 'Segmentation/UnsupervisedGlassDetection/log/', flush_secs=120)
    for _ in range(epoch - start_epoch):
        writer.add_scalar(tag='lr_epoch/train',
                          scalar_value=optim.param_groups[0]['lr'],
                          global_step=_ + start_epoch + 1)
        loss_now = train_one_epoch(_=_ + start_epoch, model=model, optimizer=optim, criterion=loss_fn,
                                   data_loader=train_loader, device=device, all_epoch=epoch, writer=writer,
                                   lr=optim.param_groups[0]['lr'])
        writer.add_scalar(tag='loss_epoch/train',
                          scalar_value=loss_now,
                          global_step=_ + start_epoch + 1)
        if (_ + start_epoch + 1) >= 10:
            if (loss_now < loss_temp) or (_ + start_epoch % 5 == 0):
                is_best = False
                if loss_now < loss_temp:
                    is_best = True
                loss_temp = loss_now
                save_checkpoint(
                    {
                        "initial_lr": lr,
                        "epoch": _ + start_epoch + 1,
                        "loss": loss_now,
                        "state_dict": model.module.state_dict(),
                        "optimizer": optim.state_dict(),
                        "scheduler": scheduler.state_dict(),
                    },
                    is_best=is_best,
                    filename=proj_path + "Segmentation/UnsupervisedGlassDetection/checkpoints/checkpoint_{:04d}.pth".format(
                        _ + start_epoch + 1),
                )
        scheduler.step()
    writer.close()


if __name__ == '__main__':
    train_multi_gpu(checkpoint=None)  # MLP head layers = 3
