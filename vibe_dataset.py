import numpy as np
import torchvision.transforms as transforms
import torch
import torch.utils.data as data
import os
import os.path as osp
import joblib
from encoder_modules import Encoder

class VIBEDatset(data.Dataset):
    def __init__(self, opt, root):
        super(VIBEDatset, self).__init__()
        self.opt = opt
        self.root = root#self.opt.vvt_vibe_root

        self.data_list = []
        for root, dirs, files in os.walk(osp.join(self.root, "VIBE")):
            for file in files:
                if file.endswith(".pkl"):
                    self.data_list.append(osp.join(self.root, root, file))

        # alphabetize to make sure it is same order




    # implement frame id parameter
    def __getitem__(self, index):
        frame_id = 100
        vibe_fname = self.data_list[index]
        output = joblib.load(vibe_fname)[1]

        frame_ids = output['frame_ids']
        mesh_verts = output['verts'] #[frame_id] # (1, 6890, 3)
        pose_params = output['pose']  # [frame_id] # (1, 72)
        body_shape_params = output['betas']  # [frame_id] # (1, 10)
        joints_3d = output['joints3d']  # [frame_id] # (1, 49, 3)

        #mesh_verts = np.expand_dims(mesh_verts, axis=0)
        #pose_params = np.expand_dims(pose_params, axis=0)
        #body_shape_params = np.expand_dims(body_shape_params, axis=0)


        cuda = torch.device('cuda')  # Default CUDA device

        #mesh_verts = torch.tensor(mesh_verts.ravel(), device=cuda)
        mesh_verts = torch.tensor(mesh_verts, device=cuda) # (1, 20670)
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
            mesh_verts,
            pose_params,
            body_shape_params,
            joints_3d,
        ]

        return vibe_result

    def __len__(self):
        return len(self.data_list)

def main():
    v = VIBEDatset(None, "/data_hdd/fw_gan_vvt/train")
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    dataloader = data.DataLoader(v, batch_size=1,
                            shuffle=True)
    #Encoder()
    for i_batch, sample_batched in enumerate(dataloader):
        frame_ids, mesh_verts, pose_params, body_shape_params, joints_3d = sample_batched
        print(frame_ids.size())
        print(mesh_verts.size())
        print(pose_params.size())
        print(body_shape_params.size())
        print(joints_3d.size())
        #print(frame_ids)
        for frame in range(mesh_verts.shape[1]):
            print(frame)

            mesh_vert = torch.flatten(mesh_verts[:, frame], 1, 2)
            pose_param = pose_params[:, frame]
            body_shape_param = body_shape_params[:, frame]
            joint_3d = torch.flatten(joints_3d[:, frame], 1, 2)

            print(mesh_vert.size())
            print(pose_param.size())
            print(body_shape_param.size())
            print(joint_3d.size())

            mesh_encoder = Encoder(mesh_vert.shape[1], 192 * 256).cuda()
            mesh_encoder.eval()
            out_mesh = mesh_encoder(mesh_vert)
            torch.cuda.empty_cache()

            pose_encoder = Encoder(pose_param.shape[1], 192 * 256).cuda()
            pose_encoder.eval()
            out_pose = pose_encoder(pose_param)
            torch.cuda.empty_cache()

            body_encoder = Encoder(body_shape_param.shape[1], 192 * 256).cuda()
            body_encoder.eval()
            out_body = body_encoder(body_shape_param)
            torch.cuda.empty_cache()

            joints_encoder = Encoder(joint_3d.shape[1], 192 * 256).cuda()
            joints_encoder.eval()
            out_joints = joints_encoder(joint_3d)
            torch.cuda.empty_cache()

            out_mesh = mesh_encoder(mesh_vert)
            out_pose = pose_encoder(pose_param)
            out_body = body_encoder(body_shape_param)
            out_joints = joints_encoder(joint_3d)
            print("mesh shape:", out_mesh.size())
            print("pose shape:", out_pose.size())
            print("body shape:", out_body.size())
            print("joints shape:", out_joints.size())



if __name__ == "__main__":
    main()