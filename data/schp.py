import numpy as np
import torchvision.transforms as transforms
import torch
import torch.utils.data as data
import os
import os.path as osp
import joblib
import matplotlib.pyplot as plt
import argparse
from PIL import Image

class SCHPDataset(data.Dataset):
    def __init__(self, opt):
        super(SCHPDataset, self).__init__()

        self.opt = opt
        self.root = osp.join(self.opt.ann_dataroot, self.opt.datamode, "cloth")
        #self.datamode = opt.datamode # train or test or self-defined

        self.data_list = []
        for root_, dirs, files in os.walk(self.root):
            arr = []
            for file in files:
                if file.endswith('.png'):
                    arr.append(osp.join(self.root, root_, file))
            self.data_list.append(arr)
        self.data_list.pop(0)
        self.data_list.sort()

    def __getitem__(self, index):
        schp_fnames = self.data_list[index]
        #self.file_name =
        #schp_fnames.sort()

        print("SCHP File Name", schp_fnames[0].split("/")[5])
        schp_output_list = []
        for schp_fname in schp_fnames:
            #print(schp_fname)
            #schp_output = np.load(schp_fname)['arr_0']

            schp_output = Image.open(schp_fname)
            schp_output = np.asarray(schp_output)
            #print("output:", schp_output.shape)
            #print("uniques", np.unique(schp_output))
            np.expand_dims(schp_output, axis=0)
            schp_output = torch.tensor(schp_output)
            schp_output_list.append(schp_output)

        return schp_output_list



    def __len__(self):
        return len(self.data_list)

def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="GMM")
    parser.add_argument("--gpu_ids", default="")
    parser.add_argument('-j', '--workers', type=int, default=1)
    parser.add_argument('-b', '--batch-size', type=int, default=4)

    parser.add_argument("--dataroot", default="/data_hdd/fw_gan_vvt/")
    parser.add_argument("--datamode", default="train")
    parser.add_argument("--stage", default="GMM")
    parser.add_argument("--data_list", default="train_pairs.txt")
    parser.add_argument("--fine_width", type=int, default=192)
    parser.add_argument("--fine_height", type=int, default=256)
    parser.add_argument("--radius", type=int, default=5)
    parser.add_argument("--grid_size", type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate for adam')
    parser.add_argument('--tensorboard_dir', type=str, default='tensorboard', help='save tensorboard infos')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='save checkpoint infos')
    parser.add_argument('--checkpoint', type=str, default='', help='model checkpoint for initialization')
    parser.add_argument("--display_count", type=int, default=20)
    parser.add_argument("--save_count", type=int, default=100)
    parser.add_argument("--keep_step", type=int, default=100000)
    parser.add_argument("--decay_step", type=int, default=100000)
    parser.add_argument("--shuffle", action='store_true', help='shuffle input data')

    opt = parser.parse_args()
    return opt

def main():
    opt = get_opt()
    s = SCHPDataset(opt)
    print(len(s))
    x = s[0]




    #print("first:", first_.size())


if __name__ == "__main__":
    main()