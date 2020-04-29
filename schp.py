import numpy as np
import torchvision.transforms as transforms
import torch
import torch.utils.data as data
import os
import os.path as osp
import joblib

class SCHPDatset(data.Dataset):
    def __init__(self, opt, root):
        super(SCHPDatset, self).__init__()

        self.opt = opt
        self.root = root #opt.schp_root

        #self.datamode = opt.datamode # train or test or self-defined

        self.data_list = []
        for root_, dirs, files in os.walk(osp.join(self.root, "cloth")):
            for file in files:
                if file.endswith('.npz'):
                    self.data_list.append(osp.join(self.root, root_, file))


    def __getitem__(self, index):
        schp_fname = self.data_list[index]
        schp_output = np.load(schp_fname)['arr_0']
        #print(schp_output.shape)
        np.expand_dims(schp_output, axis=0)
        schp_output = torch.tensor(schp_output)
        return schp_output



    def __len__(self):
        return len(self.data_list)

def main():
    s = SCHPDatset(None, "/data_hdd/fw_gan_vvt/train")
    #print(len(s))
    first_ = s[0]




    #print("first:", first_.size())


if __name__ == "__main__":
    main()