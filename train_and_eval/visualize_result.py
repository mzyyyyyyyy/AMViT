import sys
import os
sys.path.insert(0, os.getcwd())
import torch
import json
from argparse import Namespace

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib
import numpy as np

import warnings
warnings.filterwarnings('ignore')

from utils.config_files_utils import read_yaml
from data import get_dataloaders
from models import get_model
from utils.torch_utils import get_device, load_from_checkpoint





# 定义颜色映射
def def_color():
    # Colormap (same as in the paper)
    # 这段代码的主要目的是创建一个自定义的颜色映射，
    # 这些颜色可以用于绘制图形或图像。
    cm = matplotlib.cm.get_cmap('tab20')
    def_colors = cm.colors
    cus_colors = ['k'] + [def_colors[i] for i in range(1,12)]+['w']
    cmap = ListedColormap(colors = cus_colors, name='agri',N=12)
    return cmap

# 返回带有预训练参数的 uate 模型
def load_model(path, device, fold=1, mode='semantic'):
    """Load pre-trained model"""
    with open(os.path.join(path, 'conf.json')) as file:
        config = json.loads(file.read())

    # 这行代码的作用是将config字典中的键值对解包，
    # 并将它们作为属性添加到新的Namespace对象中。
    # 这样，你就可以通过点运算符来访问这些属性，而不是通过字典的键来访问。
    config = Namespace(**config)
    model = get_model(config, mode = mode).to(device)
    # 这里的模型可以看作是模型架构，还没有参数

    sd = torch.load(
        os.path.join(path, "Fold_{}".format(fold+1), "model.pth.tar"),
        map_location=device
        )
    model.load_state_dict(sd['state_dict'])
    
    # for param_tensor in sd:
    #     print(param_tensor, "\t", sd[param_tensor])

    # 1，torch.load() 用于加载torch.save()保存的模型；
    # 2，model.load_state_dict() 将模型的参数加载到模型中。
    # state_dict是一个将网络的每一层映射到其参数张量的Python字典对象。
    return model

# 从一个 batch 的时序 patch 中提取某个 batch 的某个时间戳的 rgb 影像
def get_rgb(x,b=0,t_show=6):
    """Gets an observation from a time series and normalises it for visualisation."""
    # 输入一个 batch 的时序 patch，得到一个 rgb 图像，
    # 这是选取的 batch 中的某个时序 patch 的某个时间戳的图像。
    im = x[b,t_show,:,:,[2,1,0]].cpu().numpy()
    # 因为 b, t 在这里都是单个索引，所以 im 的 shape 是 (3, h, w)/(h, w, 3)
    mx = im.max(axis=(0,1))
    mi = im.min(axis=(0,1))
    # mx 和 mi 都是一维数组，它们的 shape 表示为 (n, )
    im = (im - mi[None,None,:])/(mx - mi)[None,None,:]
    # mi[:,None,None]和(mx - mi)[:,None,None]中的None是用来增加维度的，
    # 使得mi和mx - mi的形状与im匹配，从而可以进行元素级别的运算。
    # im = im.swapaxes(0,2).swapaxes(0,1)
    # 这两行代码将im的维度进行了交换。(C, H, W)->(H, W, C)
    im = np.clip(im, a_max=1, a_min=0)
    # 这行代码将im中的所有值限制在0-1之间。
    # 如果im中有小于0的值，就将其设为0；如果有大于1的值，就将其设为1。
    return im

def recursive_todevice(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    elif isinstance(x, dict):
        return {k: recursive_todevice(v, device) for k, v in x.items()}
    else:
        return [recursive_todevice(c, device) for c in x]


def main():
    
    config_file = "./configs/GXData/TSViT_fold1.yaml"
    config = read_yaml(config_file)
    model_weights = 'D:\\DeepSatModels_SavedModels\\saved_models\\GXData_result_ignore_cls0_modified'
    device = 'cpu'

    # 加载模型和数据集
    dataloaders = get_dataloaders(config)
    iterator = dataloaders['test'].__iter__()
    net = get_model(config, device)
    checkpoint = model_weights
    if checkpoint:
        load_from_checkpoint(net, checkpoint, partial_restore=False, device='cpu')
    net.to(device)

    batch_size = 4


    # 二，Inference
    batch = recursive_todevice(iterator.__next__(), device)
    # 看起来，迭代器的作用是：
    # 1，定义在 dataloader 对象中；
    # 2，作为函数 recursive_todevice 的参数吐出一个 batch 的数据
    # 3，用以后续的 inference.
    with torch.no_grad():
        logits = net(batch['inputs'].to(device))
        # batch 包含四个属性：dict_keys(['inputs', 'labels', 'seq_lengths', 'unk_masks'])
        logits = logits.permute(0, 2, 3, 1)
        _, predicted = torch.max(logits.data, -1)
        labels = batch['labels']
        mask = batch['unk_masks']
        mask = mask.squeeze()
        predicted[~mask] = 0


    # 三，推理结果可视化
    size = 5
    # 展示的图片大小
    show_T = 4
    fig, axes = plt.subplots(batch_size,show_T+2,figsize=((show_T+2)*size, batch_size*size))
    # 如何创建子图——也就是把展示图片的地方分成几块

    for b in range(batch_size):
        # Plot S2 background
        im = []
        for t in range(show_T):
            t_seq = t * int((batch['inputs'].shape[1])/show_T)
            im.append(get_rgb(batch['inputs'], b=b, t_show=t_seq))
            axes[b, t].imshow(im[t])
            axes[b, t].axis('off')
            axes[0, t].set_title('S2-T{}'.format(t))

        # Plot Semantic Segmentation prediction
        axes[b,show_T].matshow(predicted[b].cpu().numpy(),
                        cmap=def_color(),
                        vmin=0,
                        vmax=12)
        axes[0,show_T].set_title('Semantic Prediction')

        # Plot GT
        axes[b,show_T+1].matshow(labels[b].cpu().numpy(),
                        cmap=def_color(),
                        vmin=0,
                        vmax=12)
        axes[0,show_T+1].set_title('GT')        

    # Class Labels
    fig, ax = plt.subplots(1,1, figsize=(3,8))
    ax.matshow(np.stack([np.arange(0, 12) for _ in range(3)], axis=1), cmap = def_color())
    ax.set_yticks(ticks = range(12))
    ax.set_xticks(ticks=[])
    plt.show()

# 已知模型和数据，生成可视化 inference 结果
if __name__ == "__main__":
    main()
