import numpy as np
import torchvision.transforms as transforms
import torch
import torch.utils.data as data
import os
import os.path as osp
import argparse
from glob import glob
import joblib
from PIL import Image
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.nn.functional as F
from skimage import draw
import json
import os.path
from glob import glob
import os.path as osp
import torchvision.transforms as transforms
import torch
from PIL import ImageDraw
from PIL import Image
from multiprocessing import set_start_method
import traceback

class VVTDataset(data.Dataset):
    def __init__(self, opt):
        super(VVTDataset, self).__init__()
        self.radius = 4.5
        self.opt = opt
        self.root = self.opt.dataroot
        self._clothes_person_dir = osp.join(self.root, "lip_clothes_person")


        self.img_w = self.opt.fine_width
        self.img_h = self.opt.fine_height

        torch.cuda.set_device(1)
        #torch.set_default_tensor_type(torch.cuda.FloatTensor)

        if opt.datamode == "train":
            self._keypoints_dir = osp.join(self.root, "lip_train_frames_keypoint")
            self._frames_dir = osp.join(self.opt.ann_dataroot, self.opt.datamode, "train_frames")
        else:
            self._keypoints_dir = osp.join(self.root, "lip_test_frames_keypoint")
            self._frames_dir = osp.join(self.opt.ann_dataroot, self.opt.datamode, "test_frames")


        self._schp_dir = osp.join(self.opt.ann_dataroot, self.opt.datamode, "cloth")
        self._vibe_dir = osp.join(self.opt.ann_dataroot, self.opt.datamode, "VIBE")
        self._densepose_dir = osp.join(self.opt.ann_dataroot, self.opt.datamode, "densepose_output")
        self._keypoints = glob('{}/**/*.json'.format(self._keypoints_dir))

        #print(type(self.keypoints), len(self.keypoints))

        self.keypoints = sorted(glob(f"{self._keypoints_dir}/**/*.json"))
        """self.keypoints = []
        for root_, dirs, files in os.walk(self._keypoints_dir):
            arr = []
            for file in files:
                if file.endswith('.json'):
                    arr.append(osp.join(self.root, root_, file))
            self.keypoints.append(arr)
        self.keypoints.pop(0)
        self.keypoints.sort()
        print("length keypoints", len(self.keypoints))"""
        assert len(self.keypoints) > 0
        #p  self.to_tensor = transforms.Compose([transforms.ToTensor()])



    def get_target_frame(self, index):
        _pose_name = self.keypoints[index]
        _pose_name = _pose_name.replace("_keypoints.json", ".png")
        just_folder_and_file = _pose_name.split("/")[-2:]
        # print("VVT Folder Name", just_folder_and_file[0])
        frame_path = osp.join(self._frames_dir, *just_folder_and_file)
        frame_image = Image.open(frame_path)
        # frame = self.to_tensor(frame_image)
        frame = transforms.functional.to_tensor(frame_image)
        frame = frame.type(torch.FloatTensor)
        frame = self._pad_width_up(frame)
        assert frame.is_floating_point(), "is floating point: " + str(
            frame.is_floating_point()) + "is cuda" + str(frame.is_cuda)



        return frame

    def get_schp_frame(self, index):
        _schp_name = self.keypoints[index]

        _schp_name = _schp_name.replace("_keypoints.json", ".png")
        just_folder_and_file = _schp_name.split("/")[-2:]
        #print("SCHP Folder Name", just_folder_and_file[0])
        try:
            frame_path = osp.join(self._schp_dir, *just_folder_and_file)
            frame_image = Image.open(frame_path)
            frame_np = np.asarray(frame_image)
            frame = torch.from_numpy(frame_np)
            #print("schp", frame.size())
        except FileNotFoundError as e:
            print("schp traceback", e.__traceback__)
            #traceback.print_tb(e.__traceback__)
            frame = torch.zeros(256, 192)

        frame = frame.type(torch.FloatTensor)
        frame = self._pad_width_up(frame)
        assert frame.is_floating_point(), "is floating point: " + str(frame.is_floating_point()) + "is cuda" + str(frame.is_cuda)

    
    
        return frame

    def get_densepose_frame(self, index):

        _densepose_name = self.keypoints[index]

        _densepose_name = _densepose_name.replace("keypoints.json", "IUV.png")
        just_folder_and_file = _densepose_name.split("/")[-2:]
        #print("Densepose Folder Name", just_folder_and_file[0])
        frame_path = osp.join(self._densepose_dir, *just_folder_and_file)
        try:
            frame_image = Image.open(frame_path)
            frame_np = np.asarray(frame_image)
            frame = torch.from_numpy(frame_np)
            frame = frame.permute(2, 0, 1)
            frame = frame.type(torch.FloatTensor)
            frame = self._pad_width_up(frame)
        except FileNotFoundError as e:
            print(e.__traceback__)
            #if os.path.exists(os.path.join(self._schp_dir, *_densepose_name.replace("IUV.png", ".png").split("/")[-2:])):
            #print("frame:", index)
            frame = torch.zeros(3, 256, 256)



        #assert 1 == 0, frame.shape

        assert frame.is_floating_point(), "is floating point: " + str(frame.is_floating_point()) + "is cuda" + str(frame.is_cuda)


        return frame
    """
    def get_vibe_vid(self, index):
        _vibe_name = self.keypoints[index]
        just_folder = _vibe_name[0].split("/")[-2:-1]
        frame_path = osp.join(self._vibe_dir, *just_folder, "vid", "vibe_output.pkl")

        vibe_output = joblib.load(frame_path)[1]

        #print("VIBE File Name", frame_path.split("/")[5])
        # print("VIBE File Name", vibe_fname)
        frame_ids = vibe_output['frame_ids']
        pose_params = vibe_output['pose']  # [frame_id] # (1, 72)
        body_shape_params = vibe_output['betas']  # [frame_id] # (1, 10)
        joints_3d = vibe_output['joints3d']  # [frame_id] # (1, 49, 3)

        #cuda = torch.device('cuda')  # Default CUDA device
        pose_params = torch.tensor(pose_params)  # (1, 72)
        body_shape_params = torch.tensor(body_shape_params)  # (1, 10)
        joints_3d = torch.tensor(joints_3d)  # (1, 147)
        assert pose_params.is_floating_point() and pose_params.is_cuda, "is floating point: " + str(pose_params.is_floating_point()) + "is cuda" + str(pose_params.is_cuda)
        assert body_shape_params.is_floating_point() and body_shape_params.is_cuda, "is floating point: " + str(body_shape_params.is_floating_point()) + "is cuda" + str(body_shape_params.is_cuda)
        assert joints_3d.is_floating_point() and joints_3d.is_cuda, "is floating point: " + str(joints_3d.is_floating_point()) + "is cuda" + str(joints_3d.is_cuda)

        vibe_result = [
            frame_ids,
            pose_params,
            body_shape_params,
            joints_3d
        ]

        return vibe_result
    """
    def get_input_person_pose(self, index, target_width):
        """from cp-vton, loads the pose as white squares
        returns pose map, image of pose map
        """
        # load pose points
        _pose_name = self.keypoints[index]


        with open(_pose_name, 'r') as f:
            pose_label = json.load(f)
            pose_data = pose_label['people'][0]['pose_keypoints']
            pose_data = np.array(pose_data)
            pose_data = pose_data.reshape((-1, 3))

        point_num = pose_data.shape[0]  # how many pose joints
        assert point_num == 18, "should be 18 pose joints for guidedpix2pix"
        # construct an N-channel map tensor with -1
        pose_map = torch.zeros(point_num, self.img_h, self.img_w) - 1
        im_pose = torch.zeros(self.img_h, self.img_w)

        # draw a circle around the joint on the appropriate channel
        for i in range(point_num):
            pointx = pose_data[i, 0]
            pointy = pose_data[i, 1]
            if pointx > 1 and pointy > 1:
                rr, cc = draw.circle(pointy, pointx, self.radius, shape=(self.img_h, self.img_w))
                pose_map[i, rr, cc] = 1
                im_pose[rr, cc] = 1

        # add padding to the w/ h/
        pose_map = self._pad_width_up(pose_map, value=-1)# make the image 256x256
        im_pose = self._pad_width_up(im_pose)
        #assert all(i == -1 or i == 1 for i in torch.unique(pose_map)), f"{torch.unique(pose_map)}"
        assert pose_map.is_floating_point(), "is floating point: " + str(pose_map.is_floating_point()) + "is cuda" + str(pose_map.is_cuda)
        assert pose_map.dim() == 3, str(pose_map.size())


        return pose_map, im_pose

    def _get_input_person_path_from_index(self, index):
        """ Returns the path to the person image file that is used as input """
        pose_name = self._keypoints[index]
        person_id = pose_name.split("/")[-2]
        folder = osp.join(self._clothes_person_dir, person_id)
        #print(folder)

        files = os.listdir(folder)
        person_image_name = [f for f in files if f.endswith(".png")][0]
        #assert person_image_name.endswith(".png"), f"person images should have .png extensions: {person_image_name}"
        return osp.join(folder, person_image_name)

    def _get_input_cloth_path_from_index(self, index):
        """ Returns the path to the person image file that is used as input """
        pose_name = self._keypoints[index]
        person_id = pose_name.split("/")[-2]
        folder = osp.join(self._clothes_person_dir, person_id)
        #print(folder)

        files = os.listdir(folder)
        #print(files)
        person_image_name = [f for f in files if f.endswith(".jpg")][0]
        #print(person_image_name)
        #assert person_image_name.endswith(".jpg"), f"person images should have .png extensions: {person_image_name}"
        return osp.join(folder, person_image_name)

    def _pad_width_up(self, tensor, value=0):
        original = self.img_w
        new = self.img_h
        if original > new:
            raise ValueError("This function can only pad up if the original size is smaller than the new size")
        pad = (new - original) // 2
        new_tensor = F.pad(tensor, (pad, pad), value=value)
        return new_tensor

    def get_input_person(self, index):
        """An index specifies the keypoint; get the """
        pers_image_path = self._get_input_person_path_from_index(index)
        person_image = Image.open(pers_image_path)
        #plt.imshow(person_image)
        #person_tensor = self.to_tensor(person_image)
        person_tensor = transforms.functional.to_tensor(person_image)
        person_tensor = person_tensor.type(torch.FloatTensor)
        person_tensor = self._pad_width_up(person_tensor)
        assert person_tensor.is_floating_point(), "is floating point: " + str(person_tensor.is_floating_point()) + "is cuda" + str(person_tensor.is_cuda)

        return person_tensor

    def get_input_cloth(self, index):
        pers_cloth_path = self._get_input_cloth_path_from_index(index)
        #print(pers_cloth_path)
        cloth_img = Image.open(pers_cloth_path)
        #plt.imshow(cloth_image)
        #cloth_tensor = self.to_tensor(cloth_image)
        #cloth_tensor = transforms.functional.to_tensor(cloth_image)
        """cloth_tensor = transforms.functional.to_tensor(cloth_image)
        cloth_tensor = cloth_tensor.type(torch.cuda.FloatTensor)

        cloth_img = transforms.functional.to_pil_image(cloth_tensor)  # Image.fromarray(cloth_np).convert("L")
        cloth_img = transforms.functional.to_grayscale(cloth_img)"""
        cloth_np = np.array(cloth_img)
        #print(cloth_np.shape)
        cloth_tensor = torch.from_numpy(cloth_np)
        cloth_tensor = cloth_tensor.type(torch.FloatTensor)
        cloth_tensor = cloth_tensor.permute(2, 0, 1)
        cloth_mask = np.where(cloth_np > 240, 0, 255)

        cloth_mask = torch.from_numpy(cloth_mask)
        cloth_mask = cloth_mask.type(torch.FloatTensor)
        cloth_mask = cloth_mask.permute(2, 0, 1)

        #cloth_tensor = torch.unsqueeze(cloth_tensor, 0)
        #cloth_mask = torch.unsqueeze(cloth_mask, 0)
        #cloth_mask = torch.stack((cloth_mask, cloth_mask, cloth_mask))
        #print(cloth_mask.size())
        cloth_tensor = self._pad_width_up(cloth_tensor)

        cloth_mask = self._pad_width_up(cloth_mask)
        #assert 1 == 0, cloth_mask.size()
        assert cloth_tensor.is_floating_point(), "is floating point: " + str(cloth_tensor.is_floating_point()) + "is cuda" + str(cloth_tensor.is_cuda)
        assert cloth_mask.is_floating_point(), "is floating point: " + str(cloth_mask.is_floating_point()) + "is cuda" + str(cloth_mask.is_cuda)

        return cloth_tensor, cloth_mask

    def generate_head(self, image_target, schp):


        image = image_target.cpu().numpy()
        cloth_seg = schp.cpu().numpy()
        FACE = 13
        HAIR = 2
        out = np.where(cloth_seg == FACE, 256, 0)
        out1 = np.where(cloth_seg == HAIR, 256, 0)
        mask = out + out1


        mask = np.expand_dims(mask, axis=0)
        #print(image.shape, mask.shape)
        mask = np.vstack((mask, mask, mask))
        head = np.where(image < mask, image, 255)
        head = torch.from_numpy(head)
        head = head.type(torch.FloatTensor)

        assert head.is_floating_point(), "is floating point: " + str(head.is_floating_point()) + "is cuda" + str(head.is_cuda)



        return head

    def generate_body_shape(self, image_target, schp):


        image = image_target.cpu().numpy()
        cloth_seg = schp.cpu().numpy()


        mask = np.where(cloth_seg == 0, 256, 0)
        mask = np.expand_dims(mask, 0)
        mask = np.vstack((mask, mask, mask))
        body_shape = np.where(image < mask, 0, 255)
        body_shape = torch.from_numpy(body_shape)
        body_shape = body_shape.type(torch.FloatTensor)
        assert body_shape.is_floating_point(), "is floating point: " + str(body_shape.is_floating_point()) + "is cuda" + str(body_shape.is_cuda)



        return body_shape

    def __getitem__(self, index):

        """
        Returns: <a dict> {
            'input': input,
            "cloth": cloth,
            "cloth_mask": cloth_mask,
            "guide_path": self.keypoints[index],
            'guide': guide,
            'target': target,
            "schp": schp,
            "vibe": vibe,
            "heads": heads,
            "body_shapes": body_shapes
        }
        """
        #print("index:", index)
        image = self.get_input_person(index)  # (3, 256, 256)
        cloth, cloth_mask = self.get_input_cloth(index)   # (3, 256, 256)


        #in index video, get keypoints


        image_target = self.get_target_frame(index)
        #print("image target", len(image_target))
        schp = self.get_schp_frame(index)
        try:
            pose_target, im_pose = self.get_input_person_pose(index, target_width=256)  # (18, 256, 256)
            im_pose = im_pose.unsqueeze(0)
            #print(len(im_pose), im_pose[0].size(), len(pose_target), pose_target[0].size())
        except IndexError as e:
            print("pose traceback", e.__traceback__)
            pose_target = torch.zeros(18, 256, 256).to(device=torch.device('cpu'))
            im_pose = torch.zeros(1, 256, 256).to(device=torch.device('cpu'))

        #pose_target = pose_target.type(torch.FloatTensor)
        #im_pose = im_pose.type(torch.FloatTensor)

        #print("schp", len(schp))
        if self.opt.vibe:
            vibe = self.get_vibe_vid(index)
        head = self.generate_head(image_target, schp)
        body_shape = self.generate_body_shape(image_target, schp)
        if self.opt.densepose:
            densepose = self.get_densepose_frame(index)
        # input-guide-target
        input = (image / 255) * 2 - 1
        #print("input", len(input)) #  (3, 256, 256)

        guide = pose_target
        schp = schp.unsqueeze(0)
        #print("pose", len(guide)) #  (frames, 3, 256, 256)

        target = (image_target/255) * 2 - 1
        #print("target", len(image_target))  # (frames, 3, 256, 256)
        # Put data into [input, cloth, guide, target]
        #print("cloth_mask size", cloth_mask.size())

        #assert cloth_mask.dim() == 4

        vvt_result = {
            'input': input.to(device=torch.device('cpu')),
            "cloth": cloth.to(device=torch.device('cpu')),
            "cloth_mask": cloth_mask.to(device=torch.device('cpu')),
            "guide_path": self.keypoints[index],
            'guide': guide.to(device=torch.device('cpu')),
            "im_pose": im_pose.to(device=torch.device('cpu')),
            'target': target.to(device=torch.device('cpu')),
            "schp": schp.to(device=torch.device('cpu')),
            #"vibe": vibe,
            "head": head.to(device=torch.device('cpu')),
            "body_shape": body_shape.to(device=torch.device('cpu'))
        }

        if self.opt.vibe:
            vvt_result['vibe'] = vibe
        if self.opt.densepose:
            vvt_result['densepose'] = densepose.to(device=torch.device('cpu'))

        for key in vvt_result.keys():
            value = vvt_result[key]
            print("yup", key, type(value), type(value[0]))
            if isinstance(value[0], torch.Tensor):
                print("yup", value[0].size())


        return vvt_result

    def __len__(self):
        return len(self.keypoints)

class CPDataLoader(object):
    def __init__(self, opt, dataset):
        super(CPDataLoader, self).__init__()

        if opt.shuffle:
            train_sampler = torch.utils.data.sampler.RandomSampler(dataset)
        else:
            train_sampler = None

        self.data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=opt.batch_size,
            shuffle=(train_sampler is None),
            num_workers=opt.workers,
            #pin_memory=True,
            sampler=train_sampler,
        )
        self.dataset = dataset
        self.data_iter = self.data_loader.__iter__()

    def next_batch(self):
        try:
            batch = self.data_iter.__next__()
        except StopIteration:
            self.data_iter = self.data_loader.__iter__()
            batch = self.data_iter.__next__()

        return batch

    def __len__(self):
        return len(self.data_loader)

def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="GMM")
    parser.add_argument("--gpu_ids", default="")
    parser.add_argument('-j', '--workers', type=int, default=1)
    parser.add_argument('-b', '--batch-size', type=int, default=4)

    parser.add_argument("--dataroot", default="/data_hdd/vvt_competition/")
    parser.add_argument("--ann_dataroot", default="/data_hdd/fw_gan_vvt/")
    parser.add_argument("--datamode", default="train")
    parser.add_argument("--stage", default="GMM")
    parser.add_argument("--data_list", default="train_pairs.txt")
    parser.add_argument("--fine_width", type=int, default=192)
    parser.add_argument("--fine_height", type=int, default=256)
    parser.add_argument("--radius", type=int, default=5)

    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate for adam')
    parser.add_argument('--tensorboard_dir', type=str, default='tensorboard', help='save tensorboard infos')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='save checkpoint infos')
    parser.add_argument('--checkpoint', type=str, default='', help='model checkpoint for initialization')
    parser.add_argument("--display_count", type=int, default=20)
    parser.add_argument("--save_count", type=int, default=100)
    parser.add_argument("--keep_step", type=int, default=100000)
    parser.add_argument("--decay_step", type=int, default=100000)
    parser.add_argument("--shuffle", action='store_true', help='shuffle input data')
    parser.add_argument("--grid_size", type=int, default=5)
    parser.add_argument("--vibe", type=int, default=0)
    parser.add_argument("--densepose", type=int, default=0)

    opt = parser.parse_args()
    return opt

def collate_fn_padd(batch):
    '''
    Padds batch of variable length

    note: it converts things ToTensor manually here since the ToTensor transform
    assume it takes in images rather than arbitrary tensors.
    '''
    ## get sequence lengths
    print("collate_fn_padd:", batch, type(batch), batch.size(), batch.type())
    cuda = torch.device('cuda')
    lengths = torch.tensor([ t.shape[0] for t in batch ]).to(cuda)
    ## padd
    batch = [ torch.Tensor(t).to(cuda) for t in batch ]
    batch = torch.nn.utils.rnn.pad_sequence(batch)
    ## compute mask
    mask = (batch != 0).to(cuda)
    print("collate_fn_padd:", batch, type(batch), batch.size(), batch.type())
    return batch, lengths, mask

def main():
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass
    opt = get_opt()

    vvt = VVTDataset(opt)
    for i in range(len(vvt)):
        try:
            pose_target = vvt.get_input_person_pose(i, target_width=256)
        except IndexError as e:
            print(e.__traceback__)
            pose_target = torch.zeros(18, 256, 256)

        for pose in pose_target:

            try:
                assert pose.dim() == 3, str(pose.size()) + "\n" + str(pose) + "\n" + str(torch.unique(pose))
            except AssertionError as e:
                print(torch.unique(pose))
                if torch.unique(pose).item() == 0:
                    pose = torch.zeros(18, 256, 256)
                else:
                    raise
            print("Pose Target:", pose.type(), pose.size())




if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    main()
