import os
import logging
import time
import glob

import numpy as np
import tqdm
import torch
import torch.utils.data as data
from functions.denoising import generalized_steps,FGDM_steps,FGDM_steps_normal_noise
from models.diffusion import Model
from models.ema import EMAHelper
from functions import get_optimizer
from functions.losses import loss_registry, loss_registry_FGDM
from datasets import get_dataset, data_transform, inverse_data_transform
from functions.ckpt_util import get_ckpt_path
import random
import torchvision.utils as tvu
from torchvision import transforms
import cv2
import torchvision
import random
import colorednoise as cn

def pink_noise_like(shape):
    # 生成长度为 np.prod(shape) 的粉红噪声序列
    noise_sequence = cn.powerlaw_psd_gaussian(1, np.prod(shape))
    # 将序列转换为输入形状
    return noise_sequence.reshape(shape)

def blue_noise_like(shape):
    # 生成长度为 np.prod(shape) 的蓝噪声序列
    noise_sequence = cn.powerlaw_psd_gaussian(-1, np.prod(shape))
    # 将序列转换为输入形状
    return noise_sequence.reshape(shape)



class CBCTDataset_FGDM(data.Dataset):
    def __init__(self, path, size=None, test=None):
        self.img = os.listdir(path)
        random.shuffle(self.img)
        self.size = size
        self.path = path
        self.test = test

    def __getitem__(self, item):
        img_num = len(self.img)
        item = int(item % img_num)

        imagename = os.path.join(self.path, self.img[item])
        #print(imagename)
        npimg = cv2.imread(imagename)
        npimg = np.transpose(npimg, (2, 0, 1))[0]

        nplabs = npimg

        npimg = npimg / 255
        

        threshold_random = random.random()
        bilateralFilter_random = random.randint(1, 30)
        

        nplabs = cv2.bilateralFilter(nplabs, 10, bilateralFilter_random, bilateralFilter_random)
        nplabs = np.uint8(nplabs)

        x = cv2.Sobel(nplabs, cv2.CV_16S, 1, 0)
        y = cv2.Sobel(nplabs, cv2.CV_16S, 0, 1)
        absX = cv2.convertScaleAbs(x)
        absY = cv2.convertScaleAbs(y)
        nplabs = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
        
        nplabs[nplabs < threshold_random] = 0
        
        nplabs = (nplabs - nplabs.min() +1e-12) / (nplabs.max() - nplabs.min() +1e-8)
        npimg = npimg.astype(np.float32)
        nplabs = nplabs.astype(np.float32)

        resize = transforms.Resize([self.size, self.size])

        npimg = torch.from_numpy(np.expand_dims(npimg, 0))
        nplabs = torch.from_numpy(np.expand_dims(nplabs, 0))

        npimg = resize(npimg)
        nplabs = resize(nplabs)

        return npimg, nplabs


    def __len__(self):
        size = len(self.img)
        return size


class Dataset_FGDM_test(data.Dataset):
    def __init__(self, path_low, path_high, size=None, test=None):
        self.img = os.listdir(path_low)

        total_num = len(self.img)
        fraction = 1
        rank = 0

        num = int(total_num // fraction)
        if rank < fraction - 1:
            self.img = self.img[rank * num:(rank + 1) * num]
        else:
            self.img = self.img[rank * num:]


        self.size = size
        self.path_low = path_low
        self.path_high = path_high

    def __getitem__(self, item):

        imagename_low = os.path.join(self.path_low, self.img[item])
        #print(imagename)
        npimg = cv2.imread(imagename_low)
        npimg = np.transpose(npimg, (2, 0, 1))[0]
        npimg = npimg / 255
        
        imagename_high = os.path.join(self.path_high, self.img[item])
        npimg_high = cv2.imread(imagename_high)
        npimg_high = np.transpose(npimg_high, (2, 0, 1))[0]
        npimg_high = npimg_high / 255

        npimg = npimg.astype(np.float32)
        npimg_high = npimg_high.astype(np.float32)


        resize = transforms.Resize([self.size, self.size])

        npimg = torch.from_numpy(np.expand_dims(npimg, 0))
        npimg_high = torch.from_numpy(np.expand_dims(npimg_high, 0))


        return npimg, npimg_high, self.img[item]


    def __len__(self):
        size = len(self.img)
        return size
    



def torch2hwcuint8(x, clip=False):
    if clip:
        x = torch.clamp(x, -1, 1)
    x = (x + 1.0) / 2.0
    return x


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


class FGDM(object):
    def __init__(self, args, config, device=None, normal_noise = False):
        self.args = args
        self.config = config
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device
        self.normal_noise = normal_noise

        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0
        )
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
            # torch.cat(
            # [posterior_variance[1:2], betas[1:]], dim=0).log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()

    def train(self):
        os.environ["log_path"] = self.args.log_path
        args, config = self.args, self.config
        tb_logger = self.config.tb_logger 
        dataset = CBCTDataset_FGDM(path='/data/maia/yzhang/Yunxiang/data/Synth/brain/train/CT_new', size=config.data.image_size,test=False) 
        train_loader = data.DataLoader(
            dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
        )
        model = Model(config)

        model = model.to(self.device)
        model = torch.nn.DataParallel(model)

        optimizer = get_optimizer(self.config, model.parameters())

        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(model)
        else:
            ema_helper = None

        start_epoch, step = 0, 0
        if self.args.resume_training:
            states = torch.load(os.path.join(self.args.log_path, "ckpt.pth"))
            model.load_state_dict(states[0])

            states[1]["param_groups"][0]["eps"] = self.config.optim.eps
            optimizer.load_state_dict(states[1])
            start_epoch = states[2]
            step = states[3]
            if self.config.model.ema:
                ema_helper.load_state_dict(states[4])

        for epoch in range(start_epoch, self.config.training.n_epochs):
            data_start = time.time()
            data_time = 0
            for i, (x, y) in enumerate(train_loader):
                n = x.size(0)
                data_time += time.time() - data_start
                model.train()
                step += 1

                x = x.to(self.device)
                y = y.to(self.device)

                if random.random() > 0.5:
                    y = y*0 -1 

                x = data_transform(self.config, x)

                if self.normal_noise:
                    e = torch.randn_like(x)
                else:
                    e = torch.tensor(blue_noise_like(x.size()), dtype=torch.float32).to(self.device)


                b = self.betas

                # antithetic sampling
                t = torch.randint(
                    low=0, high=self.num_timesteps, size=(n // 2 + 1,)
                ).to(self.device)
                t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]
                loss = loss_registry_FGDM[config.model.type](model, x, y, t, e, b)

                tb_logger.add_scalar("loss", loss, global_step=step)

                logging.info(
                    f"step: {step}, loss: {loss.item()}, data time: {data_time / (i+1)}"
                )

                optimizer.zero_grad()
                loss.backward()

                try:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.optim.grad_clip
                    )
                except Exception:
                    pass
                optimizer.step()

                if self.config.model.ema:
                    ema_helper.update(model)

                if step % self.config.training.snapshot_freq == 0 or step == 1:
                    states = [
                        model.state_dict(),
                        optimizer.state_dict(),
                        epoch,
                        step,
                    ]
                    if self.config.model.ema:
                        states.append(ema_helper.state_dict())

                    torch.save(
                        states,
                        os.path.join(self.args.log_path, "ckpt_{}.pth".format(step)),
                    )
                    torch.save(states, os.path.join(self.args.log_path, "ckpt.pth"))

                data_start = time.time()

                os.makedirs(os.path.join(self.args.log_path, "training_show"), exist_ok=True)
                if i % 100 ==0: 
                    torchvision.utils.save_image(x, os.path.join(self.args.log_path, 'training_show' ,f"train_{i}_x.png"),normalize=True)
                    torchvision.utils.save_image(y, os.path.join(self.args.log_path, 'training_show' ,f"train_{i}_y.png"),normalize=True)
                    


    def sample(self):
        args, config = self.args, self.config
        model = Model(self.config)
        test_dataset = Dataset_FGDM_test(path_low='/data/maia/yli/Code/FDDM/GC_UNIT/UNIT-brain/result_GC70',path_high='/data/maia/yli/Code/FDDM/GC_UNIT/UNIT-brain/high_GC' , size=config.data.image_size,test=False) 
        train_loader = data.DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.config.data.num_workers,
            drop_last=False
        )
        if not self.args.use_pretrained:
            if getattr(self.config.sampling, "ckpt_id", None) is None:
                states = torch.load(
                    os.path.join(self.args.log_path, "ckpt.pth"),
                    map_location=self.config.device,
                )
            else:
                states = torch.load(
                    os.path.join(
                        self.args.log_path, f"ckpt_{self.config.sampling.ckpt_id}.pth"
                    ),
                    map_location=self.config.device,
                )
            model = model.to(self.device)
            model = torch.nn.DataParallel(model)
            model.load_state_dict(states[0], strict=True)

            if self.config.model.ema:
                ema_helper = EMAHelper(mu=self.config.model.ema_rate)
                ema_helper.register(model)
                ema_helper.load_state_dict(states[-1])
                ema_helper.ema(model)
            else:
                ema_helper = None
        else:
            # This used the pretrained DDPM model, see https://github.com/pesser/pytorch_diffusion
            if self.config.data.dataset == "CIFAR10":
                name = "cifar10"
            elif self.config.data.dataset == "LSUN":
                name = f"lsun_{self.config.data.category}"
            else:
                raise ValueError
            ckpt = get_ckpt_path(f"ema_{name}")
            print("Loading checkpoint {}".format(ckpt))
            model.load_state_dict(torch.load(ckpt, map_location=self.device))
            model.to(self.device)
            model = torch.nn.DataParallel(model)

        model.eval()

        if self.args.fid:
            self.sample_fid(model,train_loader)
        
        

    def sample_fid(self, model,data_loader):
        config = self.config
        img_id = len(glob.glob(f"{self.args.image_folder}/*"))
        print(f"starting from image {img_id}")

        save_name = 'image_samples_500_200'
        os.makedirs(os.path.join(self.args.exp, save_name, 'result'), exist_ok=True)

        with torch.no_grad():
            for batch, (x,edge,x_name) in enumerate(data_loader):
                x_name = x_name[0]
                
                
                x = x.cuda()
                real_A = x
                edge = edge.cuda()

                x = self.sample_image(x, model, edge)
                x = inverse_data_transform(config, x)

                fake_B = x.cuda()
                os.makedirs(os.path.join(self.args.exp, save_name, 'sample'), exist_ok=True)
                os.makedirs(os.path.join(self.args.exp, save_name, 'result'), exist_ok=True)

                
                tvu.save_image(real_A, os.path.join(self.args.exp, save_name, 'sample','real_{}'.format(x_name)), normalize=False)
                tvu.save_image(fake_B, os.path.join(self.args.exp, save_name, 'result', '{}'.format(x_name)), normalize=False)
                tvu.save_image(edge, os.path.join(self.args.exp, save_name, 'sample', f"edge_{x_name}"),normalize=False)
                img_id += 1
                print(f"sampled {x_name}")

    def sample_image(self, x, model, edge ,last=True):
        try:
            skip = self.args.skip
        except Exception:
            skip = 1

        if self.args.sample_type == "generalized":

            skip = 1
            seq = range(0, self.num_timesteps, skip)

            self.normal_noise = True
            if self.normal_noise:
                print("Using normal noise")
                xs = FGDM_steps_normal_noise(x, seq, model, edge ,self.betas, eta=self.args.eta)
            else:
                xs = FGDM_steps(x, seq, model, edge ,self.betas, eta=self.args.eta)

            x = xs
        elif self.args.sample_type == "ddpm_noisy":
            if self.args.skip_type == "uniform":
                skip = self.num_timesteps // self.args.timesteps
                seq = range(0, self.num_timesteps, skip)
            elif self.args.skip_type == "quad":
                seq = (
                    np.linspace(
                        0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps
                    )
                    ** 2
                )
                seq = [int(s) for s in list(seq)]
            else:
                raise NotImplementedError
            from functions.denoising import ddpm_steps

            x = ddpm_steps(x, seq, model, self.betas)
        else:
            raise NotImplementedError
        if last:
            x = x[0][-1]
        return x

    def test(self):
        args, config = self.args, self.config
        tb_logger = self.config.tb_logger
        model = Model(self.config)
        dataset, test_dataset = get_dataset(args, config)
        train_loader = data.DataLoader(
            test_dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
        )

        if getattr(self.config.sampling, "ckpt_id", None) is None:
            states = torch.load(
                os.path.join(self.args.log_path, "ckpt.pth"),
                map_location=self.config.device,
            )
        else:
            states = torch.load(
                os.path.join(
                    self.args.log_path, f"ckpt_{self.config.sampling.ckpt_id}.pth"
                ),
                map_location=self.config.device,
            )
        model = model.to(self.device)
        model = torch.nn.DataParallel(model)
        model.load_state_dict(states[0], strict=True)

        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(model)
            ema_helper.load_state_dict(states[-1])
            ema_helper.ema(model)


        model.eval()

        if self.args.fid:
            self.sample_fid(model)
        elif self.args.interpolation:
            self.sample_interpolation(model)
        elif self.args.sequence:
            self.sample_sequence(model)
        else:
            raise NotImplementedError("Sample procedeure not defined")
