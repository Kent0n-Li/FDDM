import torch
import numpy as np
import colorednoise as cn
import torchvision
import cv2
import random

def pink_noise_like(shape):
    noise_sequence = cn.powerlaw_psd_gaussian(1, np.prod(shape))
    return noise_sequence.reshape(shape)

def blue_noise_like(shape):
    noise_sequence = cn.powerlaw_psd_gaussian(-1, np.prod(shape))
    return noise_sequence.reshape(shape)


def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a

def sde_steps(x, seq, model, noise ,b, **kwargs):
    with torch.no_grad():
        #x = noise
        x = x*2-1
        e = torch.randn_like(x)
        n = x.size(0)
        total_noise_levels = 200
        t = (torch.ones(n) * (total_noise_levels-1)).to(x.device)
        at = compute_alpha(b, t.long())

        x = x * at.sqrt() + e * (1.0 - at).sqrt()

        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x]



        for i, j in zip(reversed(seq), reversed(seq_next)):
            if i>total_noise_levels:
                continue
            print(i)
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())
            xt = xs[-1].to('cuda')

            et = model(xt, t)
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            x0_preds.append(x0_t.to('cpu'))
            c1 = (
                kwargs.get("eta", 0) * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            )
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
            xs.append(xt_next.to('cpu'))

    return xs, x0_preds
    
def FDDM_steps_normal_noise(x, seq, model, edge ,b,total_noise_levels,high_level, **kwargs):
    with torch.no_grad():

        x = x*2-1
        e = torch.randn_like(x)
        n = x.size(0)



        t = (torch.ones(n) * (total_noise_levels-1)).to(x.device)
        at = compute_alpha(b, t.long())

        x = x * at.sqrt() + e * (1.0 - at).sqrt()

        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x]


        for i, j in zip(reversed(seq), reversed(seq_next)):
            if i>total_noise_levels:
                continue
            if i<high_level:
                edge = edge*0-1
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())
            xt = xs[-1].to('cuda')

            et = model(torch.cat((xt, edge), axis=1), t)
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            x0_preds.append(x0_t.to('cpu'))
            c1 = (
                kwargs.get("eta", 0) * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            )
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            e_new = torch.randn_like(x)
            xt_next = at_next.sqrt() * x0_t + c1 * e_new + c2 * et
            xs.append(xt_next.to('cpu'))

    return xs, x0_preds
    
def FDDM_steps(x, seq, model, edge_original ,b,total_noise_levels,high_level, **kwargs):
    with torch.no_grad():
        print("FDDM_steps")
        x = x*2-1
        e = torch.tensor(blue_noise_like(x.size()), dtype=torch.float32).to(x.device)
        n = x.size(0)
        

        t = (torch.ones(n) * (total_noise_levels-1)).to(x.device)
        at = compute_alpha(b, t.long())

        x = x * at.sqrt() + e * (1.0 - at).sqrt()

        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x.to('cpu')]


        for i, j in zip(reversed(seq), reversed(seq_next)):
            edge = edge_original.clone()
            threshold = (total_noise_levels - i) / total_noise_levels
            #edge [ edge < threshold ] = 0

            
            if i>total_noise_levels:
                continue
            if i<high_level:
                edge = edge*0-1



            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())
            xt = xs[-1].to('cuda')

            et = model(torch.cat((xt, edge), axis=1), t)
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            x0_preds.append(x0_t.to('cpu'))
            c1 = (
                kwargs.get("eta", 0) * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            )
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            e_new = torch.tensor(blue_noise_like(x.size()), dtype=torch.float32).to(x.device)

            
            xt_next = at_next.sqrt() * x0_t + c1 * e_new + c2 * et
            
            
            xs.append(xt_next.to('cpu'))

    return xs, x0_preds
    
def FDDM_steps_smooth(x, seq, model, edge_original ,b,total_noise_levels,high_level, **kwargs):
    with torch.no_grad():
        print("FDDM_steps")
        x = x*2-1
        e = torch.tensor(blue_noise_like(x.size()), dtype=torch.float32).to(x.device)
        n = x.size(0)
        

        t = (torch.ones(n) * (total_noise_levels-1)).to(x.device)
        at = compute_alpha(b, t.long())

        x = x * at.sqrt() + e * (1.0 - at).sqrt()

        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x.to('cpu')]


        for i, j in zip(reversed(seq), reversed(seq_next)):
            edge = edge_original.clone()
            threshold = (total_noise_levels - i) / total_noise_levels
            edge [ edge < threshold ] = 0

            if i>total_noise_levels:
                continue
            if i<high_level:
                edge = edge*0-1

            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())
            xt = xs[-1].to('cuda')

            et = model(torch.cat((xt, edge), axis=1), t)
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            x0_preds.append(x0_t.to('cpu'))
            c1 = (
                kwargs.get("eta", 0) * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            )
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            e_new = torch.tensor(blue_noise_like(x.size()), dtype=torch.float32).to(x.device)

            xt_next = at_next.sqrt() * x0_t + c1 * e_new + c2 * et
            
            xs.append(xt_next.to('cpu'))

    return xs, x0_preds
    
      

def build_gaussian_pyramid(image, levels):
    gaussian_pyramid = [image]
    for i in range(levels):
        image = cv2.pyrDown(image)
        gaussian_pyramid.append(image)
    return gaussian_pyramid

def build_laplacian_pyramid(gaussian_pyramid):
    laplacian_pyramid = []
    for i in range(len(gaussian_pyramid) - 1):
        next_level = cv2.pyrUp(gaussian_pyramid[i + 1], dstsize=(gaussian_pyramid[i].shape[1], gaussian_pyramid[i].shape[0]))
        laplacian = cv2.subtract(gaussian_pyramid[i], next_level)
        laplacian_pyramid.append(laplacian)
    laplacian_pyramid.append(gaussian_pyramid[-1])  
    return laplacian_pyramid

def laplacian_pyramid_merge(img1, img2, levels):
    gaussian_pyramid1 = build_gaussian_pyramid(img1, levels)
    gaussian_pyramid2 = build_gaussian_pyramid(img2, levels)
    
    laplacian_pyramid1 = build_laplacian_pyramid(gaussian_pyramid1)
    laplacian_pyramid2 = build_laplacian_pyramid(gaussian_pyramid2)


    
    laplacian_pyramid_combined = []
    for index in range(len(laplacian_pyramid1)):
        if index == (len(laplacian_pyramid1) - 1):
            laplacian_pyramid_combined.append(laplacian_pyramid2[index])
            #print(index, "2")    
        else:
            laplacian_pyramid_combined.append(laplacian_pyramid1[index])
            #print(index, "1", laplacian_pyramid1[index].shape)


    image_reconstructed = laplacian_pyramid_combined[-1]
    for i in range(levels - 1, -1, -1):
        image_reconstructed = cv2.pyrUp(image_reconstructed, dstsize=(laplacian_pyramid_combined[i].shape[1], laplacian_pyramid_combined[i].shape[0]))
        image_reconstructed = cv2.add(image_reconstructed, laplacian_pyramid_combined[i])
    
    return image_reconstructed



def FDDM_steps_2noise(x, seq, model, edge_original ,b,total_noise_levels,high_level, **kwargs):
    with torch.no_grad():
        print("FDDM_steps_fused_noise")

        x = x*2-1
        e = torch.tensor(blue_noise_like(x.size()), dtype=torch.float32).to(x.device)
        n = x.size(0)
        

        t = (torch.ones(n) * (total_noise_levels-1)).to(x.device)
        at = compute_alpha(b, t.long())

        x = x * at.sqrt() + e * (1.0 - at).sqrt()

        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x.to('cpu')]
        xs_ddim = [x.to('cpu')]


        for i, j in zip(reversed(seq), reversed(seq_next)):
            edge = edge_original.clone()
            threshold = (total_noise_levels - i) / total_noise_levels
            edge [ edge < threshold ] = 0

            if i>total_noise_levels:
                continue


            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())
            xt = xs[-1].to('cuda')
            with torch.no_grad():
                et = model(torch.cat((xt, edge), axis=1), t).detach()
            
           
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            x0_preds.append(x0_t.to('cpu'))

            xt_ddim = xs_ddim[-1].to('cuda')
            with torch.no_grad():
                et_ddim = model(torch.cat((xt_ddim, edge), axis=1), t).detach()
            
            
            x0_t_ddim = (xt_ddim - et_ddim * (1 - at).sqrt()) / at.sqrt()

            x0_t_np = x0_t.squeeze(0).to('cpu').numpy()
            x0_t_ddim_np = x0_t_ddim.squeeze(0).to('cpu').numpy()

            #convert to cv2 image format (256*256*3)
            x0_t_np = np.concatenate((x0_t_np,x0_t_np,x0_t_np),axis=0)
            x0_t_ddim_np = np.concatenate((x0_t_ddim_np,x0_t_ddim_np,x0_t_ddim_np),axis=0)

            #print(x0_t_np.shape)
            #print(x0_t_ddim_np.shape)

            x0_t_np = x0_t_np.transpose(1,2,0)
            x0_t_ddim_np = x0_t_ddim_np.transpose(1,2,0)

            #print(x0_t_np.shape)

            merged_x0 = laplacian_pyramid_merge(x0_t_np.astype(np.float32), x0_t_ddim_np.astype(np.float32), 2)
            merged_x0 = torch.tensor(merged_x0, dtype=torch.float32).to(x.device)
            merged_x0 = merged_x0[:,:,0].unsqueeze(0).unsqueeze(0)

            #print(merged_x0.shape)
            


            c1 = (
                kwargs.get("eta", 0) * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            )
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            e_new = torch.tensor(blue_noise_like(x.size()), dtype=torch.float32).to(x.device)
            xt_next_ddpm = at_next.sqrt() * merged_x0 + c1 * e_new + c2 * et


            c1 = 0
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next_ddim = at_next.sqrt() * merged_x0 + c2 * et


            xs.append(xt_next_ddpm.to('cpu'))
            xs_ddim.append(xt_next_ddim.to('cpu'))

            torch.cuda.empty_cache()
            del et, et_ddim, x0_t, x0_t_ddim, e_new


    return xs, x0_preds
   


def generalized_steps(x, seq, model, b, **kwargs):
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x]
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())
            xt = xs[-1].to('cuda')
            et = model(xt, t)
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            x0_preds.append(x0_t.to('cpu'))
            c1 = (
                kwargs.get("eta", 0) * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            )
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
            xs.append(xt_next.to('cpu'))

    return xs, x0_preds


def ddpm_steps(x, seq, model, b, **kwargs):
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        xs = [x]
        x0_preds = []
        betas = b
        print(reversed(seq))
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(betas, t.long())
            atm1 = compute_alpha(betas, next_t.long())
            beta_t = 1 - at / atm1
            x = xs[-1].to('cuda')

            output = model(x, t.float())
            e = output

            x0_from_e = (1.0 / at).sqrt() * x - (1.0 / at - 1).sqrt() * e
            x0_from_e = torch.clamp(x0_from_e, -1, 1)
            x0_preds.append(x0_from_e.to('cpu'))
            mean_eps = (
                (atm1.sqrt() * beta_t) * x0_from_e + ((1 - beta_t).sqrt() * (1 - atm1)) * x
            ) / (1.0 - at)

            mean = mean_eps
            noise = torch.randn_like(x)
            mask = 1 - (t == 0).float()
            mask = mask.view(-1, 1, 1, 1)
            logvar = beta_t.log()
            sample = mean + mask * torch.exp(0.5 * logvar) * noise
            xs.append(sample.to('cpu'))
    return xs, x0_preds
