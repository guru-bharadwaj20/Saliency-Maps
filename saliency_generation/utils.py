import torchvision.transforms as transforms
import numpy as np
import torch
from torch.autograd import Variable


def to_variable(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad)


def save_gray(img, path):
    pilTrans = transforms.ToPILImage()
    pilImg = pilTrans(img)
    print('Image saved to ', path)
    pilImg.save(path)


def predict(model_rgb, model_depth, img, epoch, path):
    to_tensor = transforms.ToTensor()  # Transforms 0-255 numbers to 0 - 1.0.
    im = to_tensor(img)
    # show(im)
    inp = to_variable(im.unsqueeze(0), False)
    # print(inp.size())
    out_rgb = model_rgb(inp)
    out_depth = model_depth(inp)
    map_out_rgb = out_rgb.cpu().data.squeeze(0)
    map_out_depth = out_depth.cpu().data.squeeze(0)
    # show_gray(map_out)

    new_path_rgb = path + str(epoch) + "rgb.png"
    new_path_depth = path + str(epoch) + "depth.png"
    save_gray(map_out_rgb, new_path_rgb)
    save_gray(map_out_depth, new_path_depth)
