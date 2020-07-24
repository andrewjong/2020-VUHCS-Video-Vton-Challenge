# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import os
import time
from PIL import Image
from data.vvt_dataset import VVTDataset
from networks import GMM, UnetGenerator, VGGLoss, load_checkpoint, save_checkpoint
from encoder_modules import Encoder
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
from visualization import board_add_image, board_add_images
from multiprocessing import set_start_method
from tqdm import tqdm



def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="GMM")
    parser.add_argument("--gpu_ids", default="")
    parser.add_argument('-j', '--workers', type=int, default=0)
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
    parser.add_argument("--grid_size", type=int, default = 5)
    parser.add_argument("--vibe", type=int, default=0)
    parser.add_argument("--densepose", type=int, default=0)

    opt = parser.parse_args()
    return opt


def train_gmm(opt, vvt_loader, model, board):

    torch.cuda.set_device(1)
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    model.cuda()
    model.train()

    #vvt_loader = iter(vvt_loader)
    criterionL1 = nn.L1Loss()

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.5, 0.999))

    '''
    for each # step
        for each # batch
            from each # frame
    '''

    cuda = torch.device('cuda:1')
    print("init train_gmm, ready to get into loop")
    im_g = Image.open("grid.png")
    im_g = np.array(im_g)
    im_g = torch.from_numpy(im_g)
    im_g = im_g.type(torch.cuda.FloatTensor)
    im_g = torch.unsqueeze(im_g, 0)
    pbar = tqdm(range(opt.keep_step + opt.decay_step), unit="step")
    for step in pbar:
        print("Started step", step)
        iter_start_time = time.time()
        for i, vvt_vid in enumerate(vvt_loader):
        #vvt_vid = next(vvt_loader)
            print(i)



            input = vvt_vid['input']
            cloth = vvt_vid['cloth']#.to(cuda)
            guide_path = vvt_vid['guide_path']
            guide = vvt_vid['guide']
            im_poses = vvt_vid['im_poses']
            target = vvt_vid['target']
            schp = vvt_vid['schp']
            if opt.vibe:
                vibe = vvt_vid['vibe']
            if opt.densepose:
                densepose = vvt_vid['densepose']
            heads = vvt_vid['heads']
            body_shapes = vvt_vid['body_shapes']
            cloth_mask = vvt_vid['cloth_mask']


            # unravel next batch
            if opt.vibe:
                frame_ids, pose_params, body_shape_params, joints_3d = vibe
                frame_ids, pose_params, body_shape_params, joints_3d = frame_ids[0], pose_params[0], body_shape_params[0], joints_3d[0]
            else:
                frame_ids = [x for x in range(len(schp))]

            for index, frame in enumerate(frame_ids):



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

                vvt_frame = target[frame]
                schp_frame = schp[frame]
                head = heads[frame]
                body_shape = body_shapes[frame]
                im_pose = im_poses[frame]
                if opt.densepose:
                    densepose_frame = densepose[frame]
                try:
                    pose = guide[frame]
                except IndexError as e:
                    print(e.__traceback__)
                    pose = torch.zeros(1, 18, 256, 256)
                #print("Pose Target:", pose.size(), pose.type())

                if opt.vibe:
                    pose_param, body_shape_param, joint_3d = pose_params[index], body_shape_params[index], joints_3d[index]
                    pose_param, body_shape_param, joint_3d = pose_param.unsqueeze(0), body_shape_param.unsqueeze(0), joint_3d.unsqueeze(0)

                    joint_3d = torch.flatten(joint_3d, 1, 2)

                    pose_encoder = Encoder(pose_param.shape[1], 256 * 256).cuda()
                    pose_encoder.eval()
                    out_pose = pose_encoder(pose_param)
                    torch.cuda.empty_cache()

                    body_encoder = Encoder(body_shape_param.shape[1], 256 * 256).cuda()
                    body_encoder.eval()
                    out_body = body_encoder(body_shape_param)
                    torch.cuda.empty_cache()

                    # TODO: does this have to be different to run all three channels?

                    joints_encoder = Encoder(joint_3d.shape[1], 256 * 256).cuda()
                    joints_encoder.eval()
                    out_joints = joints_encoder(joint_3d)
                    torch.cuda.empty_cache()

                    out_pose_param = torch.reshape(out_pose, (1, 1, 256, 256))
                    out_body_shape_param = torch.reshape(out_body, (1, 1, 256, 256))
                    out_joints_3d = torch.reshape(out_joints, (1, 1, 256, 256))





                schp_frame = schp_frame.unsqueeze(1)
                '''
                Person Representation p
                ------------------------------------
                shape, from cp-vton (1, 256, 256)
                head, from cp-vton (2, 256, 256)
                pose_map, from cp-vton (18, 256, 256)
                schp_frame, from schp (c, 256, 256)
                pose_param, from vibe (c, 256, 256)
                body_shape_param, from vibe (c, 256, 256)
                joint_3d, from vibe (c, 256, 256)
                '''
                if opt.vibe:

                    out_pose_param, out_body_shape_param, out_joints_3d = out_pose_param.to(cuda), out_body_shape_param.to(
                        cuda), out_joints_3d.to(cuda)

                    for x in [body_shape, head, pose, schp_frame, out_pose_param, out_body_shape_param, out_joints_3d]:
                        while(x.dim() < 4):
                            x = torch.unsqueeze(x, 0)
                    [print(type(x), x.size(), x.dtype, x.type()) for x in
                     [body_shape, head, pose, schp_frame, out_pose_param,
                      out_body_shape_param, out_joints_3d]]

                    p = torch.cat([body_shape, head, pose, schp_frame, out_pose_param, out_body_shape_param, out_joints_3d], 1)
                elif opt.densepose:
                    assert body_shape.dim() == 4, body_shape.size()
                    assert head.dim() == 4, head.size()
                    try:
                        assert pose.dim() == 4, str(pose.size()) + "\n" + str(torch.unique(pose))
                    except AssertionError as e:
                        print(torch.unique(pose))
                        if torch.unique(pose).item() == 0:
                            pose = torch.zeros(1, 18, 256, 256)
                        else:
                            raise

                    assert schp_frame.dim() == 4, schp_frame.size()

                    [print(type(x), x.size(), x.dtype, x.type()) for x in
                     [body_shape, head, pose, schp_frame,densepose_frame]]

                    p = torch.cat([body_shape, head, pose, schp_frame, densepose_frame], 1)
                else:

                    """if pose.dim() < 4:
                        print("pose increase")
                        pose = torch.unsqueeze(pose, 0)"""
                    """[print(type(x), x.size(), x.dtype, x.type()) for x in
                     [body_shape, head, pose, schp_frame]]"""
                    assert body_shape.dim() == 4, body_shape.size()
                    assert head.dim() == 4, head.size()
                    try:
                        assert pose.dim() == 4, str(pose.size()) + "\n" + str(torch.unique(pose))
                    except AssertionError as e:
                        print(torch.unique(pose))
                        if torch.unique(pose).item() == 0:
                            pose = torch.zeros(opt.batch_size, 18, 256, 256)
                        else:
                            raise

                    assert schp_frame.dim() == 4, schp_frame.size()

                    p = torch.cat([body_shape, head, pose, schp_frame], 1)

                assert cloth.dim() == 4, cloth.size()
                grid, theta = model(p, cloth)

                warped_cloth = F.grid_sample(cloth, grid, padding_mode='border')
                warped_mask = F.grid_sample(cloth_mask, grid, padding_mode='zeros')
                warped_grid = F.grid_sample(im_g.repeat(opt.batch_size, 1, 1, 1), grid, padding_mode='zeros')



                loss = criterionL1(warped_cloth, schp_frame)
                #print("calculated loss")
                optimizer.zero_grad()
                #print("zero grad")
                torch.cuda.empty_cache()
                #print("empty cuda cache")
                loss.backward()
                #print("did backprop")
                optimizer.step()
                #print("optimizer step")

                visuals = [
                    [head, body_shape, im_pose.unsqueeze(1)],
                    [cloth, warped_cloth, schp_frame],
                    [torch.zeros(opt.batch_size, 1, 256, 256), (warped_cloth + vvt_frame) * 0.5, vvt_frame],
                ]
                flat_visuals = [item for sublist in visuals for item in sublist]
                #[print(x.size()) for x in flat_visuals]
                pbar.set_description(f"loss: {loss.item():4f}")
                if board:
                    #print("in")
                    board_add_images(board, "combine", visuals, step + 1)
                    board.add_scalar("metric", loss.item(), step + 1)
                    tqdm.write(f'step: {step + 1:8d}, loss: {loss.item():4f}')

                save_checkpoint(model, os.path.join(opt.checkpoint_dir, opt.name, 'step_%06d.pth' % (step + 1)))


def main():
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass
    opt = get_opt()
    print(opt)
    print("Start to train stage: %s, named: %s!" % (opt.stage, opt.name))

    # =====================================================================

    # create dataset

    vvt = VVTDataset(opt)
    print("VVT Dataset Created. Length of dataset", len(vvt))



    vvt_loader = torch.utils.data.DataLoader(
        vvt, batch_size=opt.batch_size, shuffle=True,
        num_workers=opt.workers)

    print("DataLoader length", len(vvt_loader))

    print("Dataloaders created.")
    # visualization
    if not os.path.exists(opt.tensorboard_dir):
        os.makedirs(opt.tensorboard_dir)
    board = SummaryWriter(log_dir=os.path.join(opt.tensorboard_dir, opt.name))

    # create model & train & save the final checkpoint
    if opt.stage == 'GMM':
        model = GMM(opt)
        if not opt.checkpoint == '' and os.path.exists(opt.checkpoint):
            load_checkpoint(model, opt.checkpoint)
        print("About to train GMM!")
        train_gmm(opt, vvt_loader, model, board)
        save_checkpoint(model, os.path.join(opt.checkpoint_dir, opt.name, 'gmm_final.pth'))
    elif opt.stage == 'TOM':
        model = UnetGenerator(25, 4, 6, ngf=64, norm_layer=nn.InstanceNorm2d)
        if not opt.checkpoint == '' and os.path.exists(opt.checkpoint):
            load_checkpoint(model, opt.checkpoint)
        train_tom(opt, train_loader, model, board)
        save_checkpoint(model, os.path.join(opt.checkpoint_dir, opt.name, 'tom_final.pth'))
    else:
        raise NotImplementedError('Model [%s] is not implemented' % opt.stage)

    print('Finished training %s, nameed: %s!' % (opt.stage, opt.name))


if __name__ == "__main__":
    main()
