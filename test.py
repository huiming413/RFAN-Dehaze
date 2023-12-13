import torch
from models.model import Dehaze
from PIL import Image
import torchvision.transforms as tfs
import torchvision.utils as vutils
import os


def test(model, img_dir, output_dir):
    for im in os.listdir(img_dir):
        print(f'\r {im}', end='', flush=True)
        haze = Image.open(img_dir + im)
        w = haze.size[1] - haze.size[1] % 4
        h = haze.size[0] - haze.size[0] % 4
        haze1 = tfs.Compose([
            tfs.ToTensor(),
            tfs.CenterCrop((w, h))
        ])(haze)[None, ::]
        haze_no = tfs.Compose([
            tfs.ToTensor(),
            tfs.CenterCrop((w, h))
        ])(haze)[None, ::]
        haze1 = haze1.to(device)
        with torch.no_grad():
            pred = model(haze1)
            ts = torch.squeeze(pred.clamp(0, 1).cpu())
            vutils.save_image(ts, output_dir + f"/" + im.split('.')[0] + '.png')


if __name__ == '__main__':
    pk = torch.load("./trained_models/its_train.pk", map_location='cpu')
    net = Dehaze()
    net.load_state_dict(pk)
    device = 'cuda'
    net = net.to(device)
    net.eval()
    test(net, "./images/", "./output/")
