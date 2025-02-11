import torch
import torchvision
import os
import time

def noise_estimation_loss(model,
                          x0: torch.Tensor,
                          t: torch.LongTensor,
                          e: torch.Tensor,
                          b: torch.Tensor, keepdim=False):
    a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    x = x0 * a.sqrt() + e * (1.0 - a).sqrt()
    output = model(x, t.float())
    if keepdim:
        return (e - output).square().sum(dim=(1, 2, 3))
    else:
        return (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0)


loss_registry = {
    'simple': noise_estimation_loss,
}



def noise_estimation_loss_FGDM(model,
                          x0: torch.Tensor,
                          edge: torch.Tensor,
                          t: torch.LongTensor,
                          e: torch.Tensor,
                          b: torch.Tensor, keepdim=False):
    a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    x = x0 * a.sqrt() + e * (1.0 - a).sqrt()
    output = model(torch.cat((x, edge), axis=1), t.float())
    

    
    curtime=int(time.time()%1000)
    if curtime%100==0:
        
        x0_from_e = (1.0 / a).sqrt() * x - (1.0 / a - 1).sqrt() * output
        
        os.makedirs('training_show', exist_ok=True)
        torchvision.utils.save_image(x0_from_e, os.path.join('training_show' ,f"result_{curtime}.png"),normalize=True)
        torchvision.utils.save_image(x0, os.path.join('training_show' ,f"input_{curtime}.png"),normalize=True)
        torchvision.utils.save_image(x, os.path.join('training_show' ,f"noise_{curtime}.png"),normalize=True)
        torchvision.utils.save_image(edge, os.path.join('training_show' ,f"edge_{curtime}.png"),normalize=True)

    if keepdim:
        return (e - output).square().sum(dim=(1, 2, 3))
    else:
        return (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0)
    



loss_registry_FGDM = {
    'simple': noise_estimation_loss_FGDM,
}
