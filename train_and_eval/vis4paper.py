import sys
import os
sys.path.insert(0, os.getcwd())
import torch
import json
from argparse import Namespace

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib
import numpy as np

import warnings
warnings.filterwarnings('ignore')

from utils.config_files_utils import read_yaml
from data import get_dataloaders
from models import get_model
from utils.torch_utils import load_from_checkpoint


from datetime import datetime

import geopandas as gpd
import torch.utils.data as tdata

import collections.abc
import re
from torch.nn import functional as F

np_str_obj_array_pattern = re.compile(r"[SaUO]")

from models import utae


class PASTIS_Dataset(tdata.Dataset):
    def __init__(
        self,
        folder,
        norm=True,
        target="semantic",
        cache=False,
        mem16=False,
        folds=None,
        reference_date="2018-09-01",
        class_mapping=None,
        mono_date=None,
        sats=["S2"],
        # 上面这些参数都是可能从外部传递进来的参数
    ):
    # 注意，接下来几十行都是__init__函数的内容，
    # 即创建该类的实例时，自动调用的函数
        """
        Pytorch Dataset class to load samples from the PASTIS dataset, for semantic and
        panoptic segmentation.
        The Dataset yields ((data, dates), target) tuples, where:
            - data contains the image time series
            - dates contains the date sequence of the observations expressed in number
              of days since a reference date
            - target is the semantic or instance target
        Args:
            folder (str): Path to the dataset
            norm (bool): If true, images are standardised using pre-computed
                channel-wise means and standard deviations.
            reference_date (str, Format : 'YYYY-MM-DD'): Defines the reference date
                based on which all observation dates are expressed. Along with the image
                time series and the target tensor, this dataloader yields the sequence
                of observation dates (in terms of number of days since the reference
                date). 
                This sequence of dates is used for instance for the positional
                encoding in attention based approaches. 
                看来semantic不用这个信息
            target (str): 'semantic' or 'instance'. Defines which type of target is
                returned by the dataloader.
                * If 'semantic' the target tensor is a tensor containing the class of
                  each pixel.
                * If 'instance' the target tensor is the concatenation of several
                  signals, necessary to train the Parcel-as-Points module:
                    - the centerness heatmap,
                    - the instance ids,
                    - the voronoi partitioning of the patch with regards to the parcels'
                      centers,
                    - the (height, width) size of each parcel
                    - the semantic label of each parcel
                    - the semantic label of each pixel
            cache (bool): If True, the loaded samples stay in RAM, default False.
            mem16 (bool): Additional argument for cache. If True, the image time
                series tensors are stored in half precision in RAM for efficiency.
                They are cast back to float32 when returned by __getitem__.
            folds (list, optional): List of ints specifying which of the 5 official
                folds to load. By default (when None is specified) all folds are loaded.
            class_mapping (dict, optional): Dictionary to define a mapping between the
                default 18 class nomenclature and another class grouping, optional.
            mono_date (int or str, optional): If provided only one date of the
                available time series is loaded. If argument is an int it defines the
                position of the date that is loaded. If it is a string, it should be
                in format 'YYYY-MM-DD' and the closest available date will be selected.
            sats (list): defines the satellites to use (only Sentinel-2 is available
                in v1.0)
        """
        super(PASTIS_Dataset, self).__init__()
        # 这句代码就是在调用父类 tdata.Dataset 的初始化方法__init__()，
        # 这通常在子类的 __init__ 方法中完成,
        # 以确保所有在 tdata.Dataset 类中定义的属性和行为
        # 都能在 PASTIS_Dataset 类中得到保留和执行。
        self.folder = folder
        # 将外部参数传递给即将被创建对象，作为它的属性
        self.norm = norm
        self.reference_date = datetime(*map(int, reference_date.split("-")))
        self.cache = cache
        self.mem16 = mem16
        self.mono_date = (
            datetime(*map(int, mono_date.split("-"))) if mono_date and "-" in mono_date 
            else int(mono_date) if mono_date else mono_date
        )
        self.memory = {}
        self.memory_dates = {}
        self.class_mapping = (
            np.vectorize(lambda x: class_mapping[x])
            if class_mapping is not None
            else class_mapping
        )
        self.target = target
        self.sats = sats




        # Get metadata
        print("Reading patch metadata . . .")
        # 读取地理信息数据，然后按照"ID_PATCH"列的值进行排序
        self.meta_patch = gpd.read_file(os.path.join(folder, "metadata.geojson"))
        # gpd 是专门处理地理信息数据的包
        self.meta_patch.index = self.meta_patch["ID_PATCH"].astype(int)
        self.meta_patch.sort_index(inplace=True)
        # 总之，gpd 对象 self.meta_patch 长这样：
        # index：每个 patch 的 ID；
        # 属性：每个 patch 的其他属性。

        self.date_tables = {s: None for s in sats}
        # {"S2": None}
        self.date_range = np.array(range(-200, 800))
        for s in sats:
            dates = self.meta_patch["dates-{}".format(s)]
            # 至于dates的具体类型，它取决于这一列中的数据类型，
            # 如果这一列中的数据是字典类型的，那么dates就是一个字典。
            # dates 的 key 应该就是 self.meta_patch 的 index.
            date_table = pd.DataFrame(
                index=self.meta_patch.index, columns=self.date_range, dtype=int
            )
            # 在这里，self.date_range 是一个包含了一系列整数的 NumPy 数组。
            # 当它被用作 pd.DataFrame 的 columns 参数时，
            # DataFrame 会为每一个 self.date_range 中的元素创建一列。
            # 也就是说，如果 self.date_range 包含了从 -200 到 599 的整数，
            # 那么新创建的 DataFrame 就会有 800 列，列名就是这些整数。
            # 这样做的目的可能是为了创建一个具有特定列名的空 DataFrame，
            # 然后在后续的代码中，根据需要向这些列中填充数据。
            for pid, date_seq in dates.items():
            # dates 的 key 应该就是 self.meta_patch 的 index.
                d = pd.DataFrame().from_dict(date_seq, orient="index")
                # 将date_seq转换为pandas DataFrame。
                # orient参数设置为"index"意味着字典的键将被用作DataFrame的索引。
                d = d[0].apply(
                    lambda x: (
                        datetime(int(str(x)[:4]), int(str(x)[4:6]), int(str(x)[6:]))
                        - self.reference_date
                    ).days
                )
                # d[0]：这是在选择DataFrame d 的第一列（列名为0）;
                # .apply(lambda x: ...)：这是在对选定的列中的每个元素应用一个函数;
                # 这段代码是在对DataFrame d 中的每个元素应用一个函数,
                # 这个函数将元素x（假设它是一个日期，格式为YYYYMMDD）转换为datetime对象，
                # 然后计算这个日期与self.reference_date之间的天数差,
                # 这个差值（以天数表示）被存储回DataFrame d。
                date_table.loc[pid, d.values] = 1
                # 这行代码的意思是在 date_table 这个 DataFrame 中，
                # 找到行标签为 pid，列标签为 d.values 的位置，然后将这些位置的值设为1。
                # 举个例子，假设 pid 是 ‘123’，d.values 是 [2, 5, 7]，
                # 那么 date_table.loc['123', [2, 5, 7]] = 1 
                # 就会将 date_table 中行标签为 ‘123’，
                # 列标签为 2、5、7 的位置的值设为 1。
            date_table = date_table.fillna(0)
            self.date_tables[s] = {
                index: np.array(list(d.values()))
                for index, d in date_table.to_dict(orient="index").items()
            }
            # 1，首先，记起来那个表达式：expression for _ in list if expression
            # 2，在 pandas 中，to_dict() 函数用于将 DataFrame 转换为字典，
            # orient 参数决定了转换的方式。当 orient='index' 时，
            # 函数会返回一个字典，其中每个键是 DataFrame 的行标签，
            # 对应的值是另一个字典，这个字典的键是列标签，值是相应的单元格值，
            # 3，d.values()就是将字典的所有值取出，最终转换成numpy数组。
        print("Done.")
        # 所以，最终每个 patch 的时序信息存储到了 self.date_tables["S2"]中
        # 字典的 key 是 patch 序号，value 是 numpy 数组，
        # 数组的 index 表示与 reference_date 相距的天数
        # 数组的值为 0 或 1，表示是否有影像。




        # Select Fold samples
        if folds is not None:
            self.meta_patch = pd.concat(
                [self.meta_patch[self.meta_patch["Fold"] == f] for f in folds]
            )
        # 将self.meta_patch更新为只包含folds中指定的Fold的数据。
        # 具体来说，它通过列表推导式创建一个新的数据框列表，
        # 每个数据框只包含Fold列等于特定值（来自folds）的行，
        # 然后使用pd.concat()将这些数据框连接成一个新的数据框。
        self.len = self.meta_patch.shape[0]
        # .shape[0]返回数据框的行数
        self.id_patches = self.meta_patch.index
        # 所以，最终 self.meta_patch 里存了某一个或者某一些 fold 的 patches 信息。





        # Get normalisation values
        if norm:
            self.norm = {}
            for s in self.sats:
                with open(
                    os.path.join(folder, "NORM_{}_patch.json".format(s)), "r"
                ) as file:
                    normvals = json.loads(file.read())
                selected_folds = folds if folds is not None else range(1, 6)
                means = [normvals["Fold_{}".format(f)]["mean"] for f in selected_folds]
                stds = [normvals["Fold_{}".format(f)]["std"] for f in selected_folds]
                self.norm[s] = np.stack(means).mean(axis=0), np.stack(stds).mean(axis=0)
                self.norm[s] = (
                    torch.from_numpy(self.norm[s][0]).float(),
                    torch.from_numpy(self.norm[s][1]).float(),
                )
        # ？？？这里需要搞清楚的就是：NORM_S2_patch.json 文件中存的数有什么意义？
        # ？？？他们是怎么算出来的？
        else:
            self.norm = None
        print("Dataset ready.")
    # 1，总而言之，__init__()函数的作用是把传递给类的参数直接
    # 或者经过处理后赋给即将创建的实例，
    # （相当于按照某种方式整理了一下数据，方便后面使用数据，也符合抽象层次的要求）
    # 使得实例的属性包含了我们想要给它的信息。
    # 2，但是，这一部分代码并没有访问原始影像数据，
    # 访问原始影像数据的工作放在了 __getitem__ 函数里。

    def __len__(self):
        return self.len
    # 在Python类中，__len__方法有特殊的含义。
    # 当你在一个对象上调用内置的len()函数时，实际上会去调用该对象的__len__方法，
    # 因此，你可以通过在类中定义__len__方法来自定义len()函数对该类实例的行为。

    def get_dates(self, id_patch, sat):
        return self.date_range[np.where(self.date_tables[sat][id_patch] == 1)[0]]
    # ？？？这个函数感觉就返回了一个数，代表 .npy 中初始影像的日期。不确定啊？
    # 这个函数返回的是一个一维数组。np.where() 函数返回一个元组，
    # 当我们使用 [0] 时，我们实际上是在提取这个元组的第一个元素，即索引数组。






    def __getitem__(self, item):
    # 1，在Python类中，__getitem__方法有特殊的含义。
    # 当你在一个对象上使用[]操作符时，实际上会去调用该对象的__getitem__方法。
    # 因此，你可以通过在类中定义__getitem__方法来自定义[]操作符对该类实例的行为
    # 2，__getitem__方法会在DataLoader中使用，
    # 当你创建一个DataLoader对象并开始遍历它时，
    # DataLoader会在每次迭代时调用__getitem__方法来获取一个样本的数据，
    # 然后将这些样本组合成一个批次，
    # 所以，你可以在__getitem__方法中定义如何获取和处理一个样本的数据。
        id_patch = self.id_patches[item]
        # ？item 一般指的是啥？
        # 就是列表内第几个元素，和数组的 index 一样

        # Retrieve and prepare satellite data
        if not self.cache or item not in self.memory.keys():
            data = {
                satellite: np.load(
                    os.path.join(
                        self.folder,
                        "DATA_{}".format(satellite),
                        "{}.npy".format(id_patch),
                    )
                ).astype(np.float32)
                for satellite in self.sats
            }  # T x C x H x W arrays
            data = {s: torch.from_numpy(a[:, :10, :, :]) for s, a in data.items()}

            if self.norm is not None:
                data = {
                    s: (d - self.norm[s][0][None, :, None, None])
                    / self.norm[s][1][None, :, None, None]
                    for s, d in data.items()
                }

            if self.target == "semantic":
                target = np.load(
                    os.path.join(
                        self.folder, "ANNOTATIONS", "{}.npy".format(id_patch)
                    )
                )
                target = torch.from_numpy(target[0].astype(int))

                if self.class_mapping is not None:
                    target = self.class_mapping(target)

            elif self.target == "instance":
                heatmap = np.load(
                    os.path.join(
                        self.folder,
                        "INSTANCE_ANNOTATIONS",
                        "HEATMAP_{}.npy".format(id_patch),
                    )
                )

                instance_ids = np.load(
                    os.path.join(
                        self.folder,
                        "INSTANCE_ANNOTATIONS",
                        "INSTANCES_{}.npy".format(id_patch),
                    )
                )
                pixel_to_object_mapping = np.load(
                    os.path.join(
                        self.folder,
                        "INSTANCE_ANNOTATIONS",
                        "ZONES_{}.npy".format(id_patch),
                    )
                )

                pixel_semantic_annotation = np.load(
                    os.path.join(
                        self.folder, "ANNOTATIONS", "TARGET_{}.npy".format(id_patch)
                    )
                )

                if self.class_mapping is not None:
                    pixel_semantic_annotation = self.class_mapping(
                        pixel_semantic_annotation[0]
                    )
                else:
                    pixel_semantic_annotation = pixel_semantic_annotation[0]

                size = np.zeros((*instance_ids.shape, 2))
                object_semantic_annotation = np.zeros(instance_ids.shape)
                for instance_id in np.unique(instance_ids):
                    if instance_id != 0:
                        h = (instance_ids == instance_id).any(axis=-1).sum()
                        w = (instance_ids == instance_id).any(axis=-2).sum()
                        size[pixel_to_object_mapping == instance_id] = (h, w)
                        object_semantic_annotation[
                            pixel_to_object_mapping == instance_id
                        ] = pixel_semantic_annotation[instance_ids == instance_id][0]

                target = torch.from_numpy(
                    np.concatenate(
                        [
                            heatmap[:, :, None],  # 0
                            instance_ids[:, :, None],  # 1
                            pixel_to_object_mapping[:, :, None],  # 2
                            size,  # 3-4
                            object_semantic_annotation[:, :, None],  # 5
                            pixel_semantic_annotation[:, :, None],  # 6
                        ],
                        axis=-1,
                    )
                ).float()

            if self.cache:
                if self.mem16:
                    self.memory[item] = [{k: v.half() for k, v in data.items()}, target]
                else:
                    self.memory[item] = [data, target]

        else:
            data, target = self.memory[item]
            if self.mem16:
                data = {k: v.float() for k, v in data.items()}

        # Retrieve date sequences
        if not self.cache or id_patch not in self.memory_dates.keys():
            dates = {
                s: torch.from_numpy(self.get_dates(id_patch, s)) for s in self.sats
            }
            if self.cache:
                self.memory_dates[id_patch] = dates
        else:
            dates = self.memory_dates[id_patch]

        if self.mono_date is not None:
            if isinstance(self.mono_date, int):
                data = {s: data[s][self.mono_date].unsqueeze(0) for s in self.sats}
                dates = {s: dates[s][self.mono_date] for s in self.sats}
            else:
                mono_delta = (self.mono_date - self.reference_date).days
                mono_date = {
                    s: int((dates[s] - mono_delta).abs().argmin()) for s in self.sats
                }
                data = {s: data[s][mono_date[s]].unsqueeze(0) for s in self.sats}
                dates = {s: dates[s][mono_date[s]] for s in self.sats}

        if self.mem16:
            data = {k: v.float() for k, v in data.items()}

        if len(self.sats) == 1:
            data = data[self.sats[0]]
            dates = dates[self.sats[0]]
        # 从字典转到 torch 格式

        return (data, dates), target, id_patch

def get_model_utae(config, mode="semantic"):
    if mode == "semantic":
        if config.model == "utae":
            model = utae.UTAE(
                input_dim=10,
                encoder_widths=config.encoder_widths,
                decoder_widths=config.decoder_widths,
                out_conv=config.out_conv,
                str_conv_k=config.str_conv_k,
                str_conv_s=config.str_conv_s,
                str_conv_p=config.str_conv_p,
                agg_mode=config.agg_mode,
                encoder_norm=config.encoder_norm,
                n_head=config.n_head,
                d_model=config.d_model,
                d_k=config.d_k,
                encoder=False,
                return_maps=False,
                pad_value=config.pad_value,
                padding_mode=config.padding_mode,
            )
        return model
    else:
        raise NotImplementedError

def pad_tensor(x, l, pad_value=0):
    padlen = l - x.shape[0]
    pad = [0 for _ in range(2 * len(x.shape[1:]))] + [0, padlen]
    return F.pad(x, pad=pad, value=pad_value)

def pad_collate(batch, pad_value=0):
    # modified default_collate from the official pytorch repo
    # https://github.com/pytorch/pytorch/blob/master/torch/utils/data/_utils/collate.py
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if len(elem.shape) > 0:
            sizes = [e.shape[0] for e in batch]
            m = max(sizes)
            if not all(s == m for s in sizes):
                # pad tensors which have a temporal dimension
                batch = [pad_tensor(e, m, pad_value=pad_value) for e in batch]
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, 0, out=out)
    elif (
        elem_type.__module__ == "numpy"
        and elem_type.__name__ != "str_"
        and elem_type.__name__ != "string_"
    ):
        if elem_type.__name__ == "ndarray" or elem_type.__name__ == "memmap":
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError("Format not managed : {}".format(elem.dtype))

            return pad_collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, collections.abc.Mapping):
        return {key: pad_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, "_fields"):  # namedtuple
        return elem_type(*(pad_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError("each element in list of batch should be of equal size")
        transposed = zip(*batch)
        return [pad_collate(samples) for samples in transposed]

    raise TypeError("Format not managed : {}".format(elem_type))


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
    model = get_model_utae(config, mode = mode).to(device)
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
    batch_size = 4
    device = 'cpu'

    # 参数 for PTSViT
    config_file = "./configs/GXData/TSViT_fold1_vis.yaml"
    config = read_yaml(config_file)
    model_weights = 'D:\\DeepSatModels_SavedModels\\saved_models\\GXData_result_mseloss_multitokens_ignore_cls0_epoch100'

    # 参数 for TSViT-先拿别的试试
    config_file_raw = "D:\\DeepSatModels_SavedModels\\saved_models\\GXData_MSELoss_ignore_result_cls0_epoch100\config_file.yaml"
    config_raw = read_yaml(config_file_raw)
    model_weights_raw = 'D:\\DeepSatModels_SavedModels\\saved_models\\GXData_MSELoss_ignore_result_cls0_epoch100'

    # 参数 for utae
    data_fold = 'D:\\PASTIS\\ALL_GXData'
    uate_weights_fold = 'D:\\PASTIS\\ALL_GXResult\\SemanticUtaeTrainOut'    

    # 加载模型和数据集 for TSViT
    dataloaders_tsvit = get_dataloaders(config_raw)
    iterator_tsvit = dataloaders_tsvit['test'].__iter__()
    tsvit = get_model(config_raw, device)
    checkpoint_tsvit = model_weights_raw
    if checkpoint_tsvit:
        load_from_checkpoint(tsvit, checkpoint_tsvit, partial_restore=False, device='cpu')
    tsvit.to(device)

    # 加载模型和数据集 for PTSViT
    dataloaders_ptsvit = get_dataloaders(config)
    iterator_ptsvit = dataloaders_ptsvit['test'].__iter__()
    ptsvit = get_model(config, device)
    checkpoint_ptsvit = model_weights
    if checkpoint_ptsvit:
        load_from_checkpoint(ptsvit, checkpoint_ptsvit, partial_restore=False, device='cpu')
    ptsvit.to(device)

    # 加载模型和数据集 for utae
    dt = PASTIS_Dataset(folder=data_fold, norm=True,
                    target='semantic', folds=[5])
    dataloader_utae = tdata.DataLoader(dt, batch_size=batch_size, shuffle=False, collate_fn=pad_collate)
    # 迭代器
    iterator_utae =  dataloader_utae.__iter__()
    # 模型
    utae = load_model(uate_weights_fold, device=device, fold=1, mode='semantic').eval()    

    with torch.no_grad():
        for (step_pvit, batch_pvit), (step_utae, batch_utae), (step_vit, batch_vit) in zip(enumerate(dataloaders_ptsvit['test']), enumerate(dataloader_utae), enumerate(dataloaders_tsvit["test"])):
            logits_ptsvit = ptsvit(batch_pvit['inputs'].to(device))
            # batch_tsvit 包含四个属性：dict_keys(['inputs', 'labels', 'seq_lengths', 'unk_masks'])
            logits_ptsvit = logits_ptsvit.permute(0, 2, 3, 1)
            _, predicted_ptsvit = torch.max(logits_ptsvit.data, -1)
            labels_ptsvit = batch_pvit['labels']
            mask_ptsvit = batch_pvit['unk_masks']
            mask_ptsvit = mask_ptsvit.squeeze()
            predicted_ptsvit[~mask_ptsvit] = 0

            (x, dates), y, id_patch = batch_utae
            logits_utae = utae(x, batch_positions=dates)
            predicted_utae = logits_utae.argmax(dim=1)
            predicted_utae[y==0] = 0

            logits_tsvit = tsvit(batch_pvit['inputs'].to(device))
            # batch_tsvit 包含四个属性：dict_keys(['inputs', 'labels', 'seq_lengths', 'unk_masks'])
            logits_tsvit = logits_tsvit.permute(0, 2, 3, 1)
            _, predicted_tsvit = torch.max(logits_tsvit.data, -1)
            labels_tsvit = batch_pvit['labels']
            mask_tsvit = batch_pvit['unk_masks']
            mask_tsvit = mask_tsvit.squeeze()
            predicted_tsvit[~mask_tsvit] = 0



            # 推理结果可视化
            size = 3
            # 展示的图片大小
            fig, axes = plt.subplots(batch_size,4,figsize=((4)*size, batch_size*size))
            # 如何创建子图——也就是把展示图片的地方分成几块

            for b in range(batch_size):
                # Plot Semantic Segmentation prediction for utae
                axes[b,0].matshow(predicted_utae[b].cpu().numpy(),
                                cmap=def_color(),
                                vmin=0,
                                vmax=12)
                axes[0,0].set_title('UTAE') 


                # Plot Semantic Segmentation prediction for tsvit
                axes[b,1].matshow(predicted_tsvit[b].cpu().numpy(),
                                cmap=def_color(),
                                vmin=0,
                                vmax=12)
                axes[0,1].set_title('TSViT')
        


                # Plot Semantic Segmentation prediction for ptsvit
                axes[b,2].matshow(predicted_ptsvit[b].cpu().numpy(),
                                cmap=def_color(),
                                vmin=0,
                                vmax=12)
                axes[0,2].set_title('PTSViT')
      

                # Plot GT
                axes[b,3].matshow(labels_ptsvit[b].cpu().numpy(),
                                cmap=def_color(),
                                vmin=0,
                                vmax=12)
                axes[0,3].set_title('GT')        

            # Class Labels
            fig, ax = plt.subplots(1,1, figsize=(3,8))
            ax.matshow(np.stack([np.arange(0, 12) for _ in range(3)], axis=1), cmap = def_color())
            ax.set_yticks(ticks = range(12))
            ax.set_xticks(ticks=[])
            plt.show()

# 已知模型和数据，生成可视化 inference 结果
if __name__ == "__main__":
    main()
