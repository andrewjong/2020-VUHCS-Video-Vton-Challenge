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


class VVTDataset(data.Dataset):
    def __init__(self, opt):
        super(VVTDataset, self).__init__()
        self.radius = 4.5
        self.opt = opt
        self.root = self.opt.dataroot
        self._clothes_person_dir = osp.join(self.root, "lip_clothes_person")


        self.img_w = self.opt.fine_width
        self.img_h = self.opt.fine_height

        if opt.datamode == "train":
            self._keypoints_dir = osp.join(self.root, "lip_train_frames_keypoint")
            self._frames_dir = osp.join(self.root, "lip_train_frames")
            self._schp_dir = osp.join(self.root)
        else:
            self._keypoints_dir = osp.join(self.root, "lip_test_frames_keypoint")
            self._frames_dir = osp.join(self.root, "lip_test_frames")

        self._schp_dir = osp.join(self.opt.ann_dataroot, self.opt.datamode, "cloth")
        self._vibe_dir = osp.join(self.opt.ann_dataroot, self.opt.datamode, "VIBE")

        self._keypoints = glob('{}/**/*.json'.format(self._keypoints_dir))

        #print(type(self.keypoints), len(self.keypoints))
        self.keypoints = []
        for root_, dirs, files in os.walk(self._keypoints_dir):
            arr = []
            for file in files:
                if file.endswith('.json'):
                    arr.append(osp.join(self.root, root_, file))
            self.keypoints.append(arr)
        self.keypoints.pop(0)
        self.keypoints.sort()
        print("length keypoints", len(self.keypoints))
        assert len(self.keypoints) > 0
        self.to_tensor = transforms.Compose([transforms.ToTensor()])

        """print("then")
        self.data_list = []

        for root_, dirs, files in os.walk(self.root):
            arr = []
            for file in files:
                if file.endswith('.png'):
                    arr.append(osp.join(self.root, root_, file))
            self.data_list.append(arr)

        self.data_list.pop(0)
        self.data_list.sort()
        #self.data_list.sort(key=lambda x: x[0].split("/")[4])"""

    def get_target_frame(self, index):
        _pose_names = self.keypoints[index]
        frames = []
        for _pose_name in _pose_names:
            _pose_name = _pose_name.replace("_keypoints.json", ".png")
            just_folder_and_file = _pose_name.split("/")[-2:]
            print("VVT Folder Name", just_folder_and_file[0])
            frame_path = osp.join(self._frames_dir, *just_folder_and_file)
            frame_image = Image.open(frame_path)
            #plt.imshow(frame_image)
            frame = self.to_tensor(frame_image)
            frame = self._pad_width_up(frame)
            frames.append(frame)

        return frames

    def get_schp_frame(self, index):
        _schp_names = self.keypoints[index]
        frames = []
        for _schp_name in _schp_names:
            #print(_schp_name)
            _schp_name = _schp_name.replace("_keypoints.json", ".png")
            just_folder_and_file = _schp_name.split("/")[-2:]
            print("SCHP Folder Name", just_folder_and_file[0])
            frame_path = osp.join(self._schp_dir, *just_folder_and_file)
            frame_image = Image.open(frame_path)
            frame_np = np.asarray(frame_image)
            #print("frame_np:", np.unique(frame_np), frame_np.shape)
            #plt.imshow(frame_image)
            frame = torch.from_numpy(frame_np)
            frame = self._pad_width_up(frame)
            #print("unique frame:", torch.unique(frame), frame.size())
            frames.append(frame)
    
    
        return frames

    def get_vibe_vid(self, index):
        _vibe_name = self.keypoints[index]

        #print(_vibe_name)
        #_vibe_name = _vibe_name.replace("_keypoints.json", ".pkl")
        just_folder = _vibe_name[0].split("/")[-2:-1]
        frame_path = osp.join(self._vibe_dir, *just_folder, "vid", "vibe_output.pkl")

        # ++++++++++++++++++++++++++++
        vibe_output = joblib.load(frame_path)[1]

        print("VIBE File Name", frame_path.split("/")[5])
        # print("VIBE File Name", vibe_fname)
        frame_ids = vibe_output['frame_ids']
        pose_params = vibe_output['pose']  # [frame_id] # (1, 72)
        body_shape_params = vibe_output['betas']  # [frame_id] # (1, 10)
        joints_3d = vibe_output['joints3d']  # [frame_id] # (1, 49, 3)

        # mesh_verts = np.expand_dims(mesh_verts, axis=0)
        # pose_params = np.expand_dims(pose_params, axis=0)
        # body_shape_params = np.expand_dims(body_shape_params, axis=0)

        cuda = torch.device('cuda')  # Default CUDA device

        # mesh_verts = torch.tensor(mesh_verts.ravel(), device=cuda)
        # mesh_verts = torch.tensor(mesh_verts, device=cuda) # (1, 20670)
        pose_params = torch.tensor(pose_params, device=cuda)  # (1, 72)
        body_shape_params = torch.tensor(body_shape_params, device=cuda)  # (1, 10)
        joints_3d = torch.tensor(joints_3d, device=cuda)  # (1, 147)

        '''print("encoder")
        x = Encoder(mesh_verts.shape[0], 256 * 192).cuda()
        print(x)
        out = x(mesh_verts)
        print(out.shape)'''

        # TODO: include encoding to get right size
        vibe_result = [
            frame_ids,
            pose_params,
            body_shape_params,
            joints_3d
        ]

        return vibe_result

    def get_input_person_pose(self, index, target_width):
        """from cp-vton, loads the pose as white squares
        returns pose map, image of pose map
        """
        # load pose points
        _pose_names = self.keypoints[index]
        pose_maps = []
        for _pose_name in _pose_names:
            with open(_pose_name, 'r') as f:
                pose_label = json.load(f)
                pose_data = pose_label['people'][0]['pose_keypoints']
                pose_data = np.array(pose_data)
                pose_data = pose_data.reshape((-1, 3))

            point_num = pose_data.shape[0]  # how many pose joints
            #assert point_num == 17, "should be 18 pose joints for guidedpix2pix"
            # construct an N-channel map tensor with -1
            pose_map = torch.zeros(point_num, self.img_h, self.img_w) - 1

            # draw a circle around the joint on the appropriate channel
            for i in range(point_num):
                pointx = pose_data[i, 0]
                pointy = pose_data[i, 1]
                if pointx > 1 and pointy > 1:
                    rr, cc = draw.circle(pointy, pointx, self.radius, shape=(self.img_h, self.img_w))
                    pose_map[i, rr, cc] = 1

            # add padding to the w/ h/
            pose_map = self._pad_width_up(pose_map, value=-1)# make the image 256x256
            #assert all(i == -1 or i == 1 for i in torch.unique(pose_map)), f"{torch.unique(pose_map)}"
            pose_maps.append(pose_map)

        return pose_maps
    def _get_input_person_path_from_index(self, index):
        """ Returns the path to the person image file that is used as input """
        pose_name = self._keypoints[index]
        person_id = pose_name.split("/")[-2]
        folder = osp.join(self._clothes_person_dir, person_id)
        print(folder)

        files = os.listdir(folder)
        person_image_name = [f for f in files if f.endswith(".png")][0]
        #assert person_image_name.endswith(".png"), f"person images should have .png extensions: {person_image_name}"
        return osp.join(folder, person_image_name)

    def _get_input_cloth_path_from_index(self, index):
        """ Returns the path to the person image file that is used as input """
        pose_name = self._keypoints[index]
        person_id = pose_name.split("/")[-2]
        folder = osp.join(self._clothes_person_dir, person_id)
        print(folder)

        files = os.listdir(folder)
        print(files)
        person_image_name = [f for f in files if f.endswith(".jpg")][0]
        print(person_image_name)
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
        person_tensor = self.to_tensor(person_image)
        person_tensor = self._pad_width_up(person_tensor)
        return person_tensor

    def get_input_cloth(self, index):
        pers_cloth_path = self._get_input_cloth_path_from_index(index)
        print(pers_cloth_path)
        cloth_image = Image.open(pers_cloth_path)
        #plt.imshow(cloth_image)
        cloth_tensor = self.to_tensor(cloth_image)
        cloth_img = transforms.functional.to_pil_image(cloth_tensor)  # Image.fromarray(cloth_np).convert("L")
        cloth_img = transforms.functional.to_grayscale(cloth_img)
        cloth_np = np.array(cloth_img)
        cloth_mask = np.where(cloth_np > 240, 0, 255)
        cloth_mask = torch.from_numpy(cloth_mask)
        cloth_mask = self._pad_width_up(cloth_mask)
        cloth_tensor = self._pad_width_up(cloth_tensor)
        return cloth_tensor, cloth_mask

    def generate_head(self, image_target, schp):
        heads = []
        for i in range(len(image_target)):
            image = image_target[i]
            cloth_seg = schp[i]
            image = image.numpy()
            cloth_seg = cloth_seg.numpy()
            FACE = 13
            HAIR = 2
            out = np.where(cloth_seg == FACE, 256, 0)
            out1 = np.where(cloth_seg == HAIR, 256, 0)
            mask = out + out1

            mask = np.expand_dims(mask, 0)
            mask = np.vstack((mask, mask, mask))
            head = np.where(image < mask, image, 255)
            head = torch.from_numpy(head)


            heads.append(head)

        return heads

    def generate_body_shape(self, image_target, schp):
        body_shapes = []
        for i in range(len(image_target)):
            image = image_target[i]
            cloth_seg = schp[i]
            image = image.numpy()
            cloth_seg = cloth_seg.numpy()


            mask = np.where(cloth_seg == 0, 256, 0)
            mask = np.expand_dims(mask, 0)
            mask = np.vstack((mask, mask, mask))
            body_shape = np.where(image < mask, 0, 255)
            body_shape = torch.from_numpy(body_shape)
            body_shapes.append(body_shape)

        return body_shapes

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
        print("index:", index)
        image = self.get_input_person(index)  # (3, 256, 256)
        cloth, cloth_mask = self.get_input_cloth(index)   # (3, 256, 256)


        #in index video, get keypoints
        try:
            pose_target = self.get_input_person_pose(index, target_width=256)  # (18, 256, 256)
        except IndexError as e:
            print(e.__traceback__)
            pose_target = torch.zeros(18, 256, 256)

        image_target = self.get_target_frame(index)
        schp = self.get_schp_frame(index)
        if self.opt.vibe:
            vibe = self.get_vibe_vid(index)
        heads = self.generate_head(image_target, schp)
        body_shapes = self.generate_body_shape(image_target, schp)


        # random fliping
        # if (self.opt.isTrain):
        #     x, x_target, pose, pose_target, mask, mask_target = self.randomFlip(x,
        #                                                                         x_target,
        #                                                                         pose,
        #                                                                         pose_target,
        #                                                                         mask,
        #                                                                         mask_target)

        # input-guide-target
        input = (image / 255) * 2 - 1
        print("input", len(input)) #  (3, 256, 256)

        guide = pose_target
        print("pose", len(guide)) #  (frames, 3, 256, 256)


        #target = (image_target / 255) * 2 - 1
        target = [(target_tensor/255) * 2 - 1 for target_tensor in image_target]
        print("target", len(image_target))  # (frames, 3, 256, 256)
        # Put data into [input, cloth, guide, target]
        print("cloth_mask size", cloth_mask.size())

        #assert cloth_mask.dim() == 4

        #plt.show()
        vvt_result = {
            'input': input,
            "cloth": cloth,
            "cloth_mask": cloth_mask,
            "guide_path": self.keypoints[index],
            'guide': guide,
            'target': target,
            "schp": schp,
            #"vibe": vibe,
            "heads": heads,
            "body_shapes": body_shapes
        }

        if self.opt.vibe:
            vvt_result['vibe'] = vibe

        return vvt_result



    def __len__(self):
        return len(self.keypoints)


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
    parser.add_argument("--vibe", type=int, default=1)

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

    print(opt)
    print("Start to train stage: %s, named: %s!" % (opt.stage, opt.name))

    # =====================================================================

    # create dataset
    cuda = torch.device('cuda')
    vvt = VVTDataset(opt)
    print("VVT Dataset Created. Length of dataset", len(vvt))
    """schp = SCHPDataset(opt)
    print("SCHP Dataset Created. Length of dataset", len(schp))
    vibe = VIBEDataset(opt)
    print("VIBE Dataset Created. Length of dataset", len(vibe))"""

    # create dataloader

    vvt_loader = torch.utils.data.DataLoader(
        vvt, batch_size=opt.batch_size, shuffle=None,
        num_workers=opt.workers, collate_fn=collate_fn_padd)
    vvt_loader = iter(vvt_loader)
    # criterion


    # optimizer

    '''
    for each # step
        for each # batch
            from each # frame
    '''

    print("init train_gmm, ready to get into loop")

    i = 0
    for step in range(opt.keep_step + opt.decay_step):
        print("Started step", step)
        vvt_vid = vvt[i]
        i += 1

        input = vvt_vid['input']
        cloth = vvt_vid['cloth'].to(cuda)
        guide_path = vvt_vid['guide_path']
        guide = vvt_vid['guide']
        target = vvt_vid['target']
        schp = vvt_vid['schp']
        if opt.vibe:
            vibe = vvt_vid['vibe']
        heads = vvt_vid['heads']
        cloth_mask = vvt_vid['cloth_mask']
        # print("cloth_mask size", cloth_mask.size())
        cloth_mask = torch.unsqueeze(cloth_mask, 0)
        # print("cloth_mask size", cloth_mask.size())
        #assert cloth_mask.dim() == 4
        body_shapes = vvt_vid['body_shapes']
        # print(len(schp))
        # print(len(input))
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print(len(input), len(guide), len(target), len(schp), len(heads), cloth_mask.size(), len(body_shapes))
        assert (len(target) == len(schp) == len(heads) == len(body_shapes))
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

        # unravel next batch
        if opt.vibe:
            frame_ids, pose_params, body_shape_params, joints_3d = vibe
            "frame_ids, pose_params, body_shape_params, joints_3d = frame_ids[0], pose_params[0], body_shape_params[0], \
                                                                   joints_3d[0]"

            print("in vibe", type(frame_ids), len(frame_ids))
            print("Basic Size as gotten from the VIBEDataset")
            print("Frame IDs:", len(frame_ids), frame_ids)
            print("Pose Parameters:", pose_params.size())
            print("Body Shape Parameters:", body_shape_params.size())
            print("Joints 3D Size:", joints_3d.size())
        else:
            frame_ids = [x for x in range(len(schp))]
            print("not in vibe", type(frame_ids))

        for index, frame in enumerate(frame_ids):
            #print("frame:", len(frame_ids), frame)
            # print("index:", index, "frame:", frame.item())

            '''
            Created in Algorithm # TODO: (need to create)
            agnostic [pose, head, shape]
            shape
            head
            cloth_mask

            Passed into Algorithm # TODO: (find where it is and use input it)
            cloth - from vvt -> (cloth)
            image - from vvt -> (input)
            parse_cloth - cloth segmentation? --> loaded through schp
            pose_image - pose points? can we generate this 18 channel map? --> loaded through vvt
            grid_image - just a png



            '''
            try:
                vvt_frame = target[frame]
            except:
                print("index:", index)
                print("frame:", frame)
                print(len(target), target[0].size())
            schp_frame = schp[frame]
            head = heads[frame]
            body_shape = body_shapes[frame]
            pose = guide[frame]
            """print("vvt_frame:", vvt_frame.size(), vvt_frame.type())
            print("schp_frame:", schp_frame.size(), schp_frame.type())
            print("head:", head.size(), head.type())
            print("body_shape:", body_shape.size(), body_shape.type())
            print("pose:", pose.size(), pose.type())"""

    """

    input = out['input']
    guide_path = out['guide_path']
    guide = out["guide"]
    target = out['target']
    print(type(out))
    print(out.keys())
    print(type(input), type(guide_path), type(guide), type(target))
    print(input.size(), len(guide_path), len(guide), len(target))"""
    #plt.imshow(A.permute(1, 2, 0))
    #plt.imshow(guide.permute(1, 2, 0))
    #plt.imshow(B.permute(1, 2, 0))

    #plt.show()

    """print([x[0] for x in vvt.data_list])
    vvt.data_list.sort()
    print("###############################################################")
    print("###############################################################")
    print("###############################################################")
    print("###############################################################")
    print([x[0] for x in vvt.data_list])"""

    """vibe = VIBEDataset(opt)
    schp = VIBEDataset(opt)
    vvt.data_list.sort()
    schp.data_list.sort()
    vibe.data_list.sort()

    print(vvt.data_list[1][0].split("/")[4])
    print(schp.data_list[1].split("/")[5])
    print(vibe.data_list[1].split("/")[5])"""




    #print(vvt.data_list)
    #.sort(key=lambda x: x[0].split("/")[4])
    """schp = SCHPDataset(opt)
    vibe = VIBEDataset(opt)


    #first_ = s[0]
    #print(vvt[0])
    vid_index = 0 # has to be from

    vvt_output = vvt[vid_index]
    schp_output = schp[vid_index]
    vibe_output = vibe[vid_index]

    print("VVT Info", len(vvt), type(vvt))
    print("SCHP Info", len(schp), type(schp))
    print("VIBE Info", len(vibe), type(vibe))

    print("VVT Info", len(vvt_output))
    print("SCHP Info", len(schp_output))
    print("VIBE Info", len(vibe_output[0]))

    frame_ids, pose_params, body_shape_params, joints_3d = vibe_output
    frame_index = 50
    offset = frame_ids[0]

    print(frame_index + offset)
    vvt_frame = vvt_output[frame_index]
    schp_frame = schp_output[frame_index]

    print("VVT Frame:", type(vvt_frame), vvt_frame.size())

    print("SCHP Frame:", type(schp_frame), schp_frame.size())
    plt.figure()

    # subplot(r,c) provide the no. of rows and columns
    f, axarr = plt.subplots(1, 2)

    # use the created array to output your multiple images. In this case I have stacked 4 images vertically
    axarr[0].imshow(vvt_frame)
    axarr[1].imshow(schp_frame)
    #plt.imshow(vibe_output[frame_index].permute(1, 2, 0))
    plt.show()
"""
    #print(len(vibe_output))
    #print(schp[0])



    #print("first:", first_.size())

if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    main()
