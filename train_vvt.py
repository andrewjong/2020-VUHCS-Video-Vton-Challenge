# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import os
import time
from PIL import Image
from cp_dataset import CPDataLoader
from data.vibe_dataset import VIBEDataset
from data.schp import SCHPDataset
from data.vvt_dataset import VVTDataset
from networks import GMM, UnetGenerator, VGGLoss, load_checkpoint, save_checkpoint
from encoder_modules import Encoder

from tensorboardX import SummaryWriter
from visualization import board_add_image, board_add_images
from multiprocessing import set_start_method



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
    parser.add_argument("--vibe", type=int, default=1)

    opt = parser.parse_args()
    return opt


def train_gmm(opt, vvt_loader, model, board):
    model.cuda()
    model.train()

    vvt_loader = iter(vvt_loader)
    # criterion
    criterionL1 = nn.L1Loss()

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 1.0 -
                                                                                    max(0,
                                                                                        step - opt.keep_step) / float(
        opt.decay_step + 1))
    '''
    for each # step
        for each # batch
            from each # frame
    '''
    cuda = torch.device('cuda')
    print("init train_gmm, ready to get into loop")
    im_g = Image.open("grid.png")
    im_g = np.array(im_g)
    im_g = torch.from_numpy(im_g)
    im_g = torch.unsqueeze(im_g, 0)
    im_g = im_g.type(torch.FloatTensor)
    im_g = im_g.to(cuda)

    for step in range(opt.keep_step + opt.decay_step):
        iter_start_time = time.time()
        vvt_vid = next(vvt_loader)
        """vibe_inputs = next(vibe_loader)
        schp_vid = next(schp_loader)"""


        # TODO: bug with order of dataloading
        # print("VVT:", type(vvt_vid), len(vvt_vid), type(vvt_vid[0]))
        """print("VIBE:", type(vibe_inputs), len(vibe_inputs), type(vibe_inputs[0]), len(vibe_inputs[0][0]), vibe_inputs[0][-1][-1].item())
        print("SCHP:", type(schp_vid), len(schp_vid), type(schp_vid[0]))"""
        input = vvt_vid['input']
        cloth = vvt_vid['cloth'].to(cuda)
        guide_path = vvt_vid['guide_path']
        guide = vvt_vid['guide']
        target = vvt_vid['target']
        schp = vvt_vid['schp']
        if opt.vibe:
            vibe = vvt_vid['vibe']
        heads = vvt_vid['heads']
        cloth_masks = vvt_vid['cloth_masks']
        body_shapes = vvt_vid['body_shapes']



        # unravel next batch
        if opt.vibe:
            frame_ids, pose_params, body_shape_params, joints_3d = vibe
            frame_ids, pose_params, body_shape_params, joints_3d = frame_ids[0], pose_params[0], body_shape_params[0], joints_3d[0]
        else:
            frame_ids = [x for x in range(len(schp))]

        for index, frame in enumerate(frame_ids):
            #print("index:", index, "frame:", frame.item())
            # which of these are generated vs which are created

            # MAJOR
            # ROADBLOCK
            # look
            # into!
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
            
            List
            
            
                         
            '''

            vvt_frame = target[frame]
            schp_frame = schp[frame]
            head = heads[frame]
            cloth_mask = cloth_masks[frame]
            body_shape = body_shapes[frame]
            pose = guide[frame]

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
            '''im = vvt_inputs['image'].cuda()
            im_pose = vvt_inputs['pose_image'].cuda()
            im_h = vvt_inputs['head'].cuda()
            shape = vvt_inputs['shape'].cuda()
            agnostic = vvt_inputs['agnostic'].cuda()
            c = vvt_inputs['cloth'].cuda()
            cm = vvt_inputs['cloth_mask'].cuda()
            im_c = vvt_inputs['parse_cloth'].cuda()
            im_g = vvt_inputs['grid_image'].cuda()'''

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

            """[print(type(x), x.size(), x.dtype, x.type()) for x in [body_shape, head, pose, schp_frame, out_pose_param,
                    out_body_shape_param, out_joints_3d]]"""

            body_shape = body_shape.type(torch.FloatTensor)

            body_shape, head, pose, schp_frame = body_shape.to(cuda), head.to(cuda), pose.to(cuda), schp_frame.to(cuda)

            if opt.vibe:
                out_pose_param, out_body_shape_param, out_joints_3d = out_pose_param.to(cuda), out_body_shape_param.to(
                    cuda), out_joints_3d.to(cuda)
                p = torch.cat([body_shape, head, pose, schp_frame, out_pose_param, out_body_shape_param, out_joints_3d], 1)
            else:
                p = torch.cat([body_shape, head, pose, schp_frame], 1)

            """[print(type(x), x.size(), x.dtype, x.type()) for x in [body_shape, head, pose, schp_frame, out_pose_param,
                                                         out_body_shape_param, out_joints_3d]]"""
            cloth_mask = cloth_mask.type(torch.FloatTensor)
            cloth_mask = cloth_mask.to(cuda)

            print("cloth mask", type(cloth_mask))
            print("finished person representation", p.size())
            grid, theta = model(p, cloth)
            print(grid, theta)
            print(type(grid), type(theta))
            print(grid.size(), theta.size())

            warped_cloth = F.grid_sample(cloth, grid, padding_mode='border')
            warped_mask = F.grid_sample(cloth_mask, grid, padding_mode='zeros')
            warped_grid = F.grid_sample(im_g, grid, padding_mode='zeros')

            """visuals = [[im_h, shape, im_pose],
                       [c, warped_cloth, im_c],
                       [warped_grid, (warped_cloth + im) * 0.5, im]]"""
            print(warped_cloth.size())
            print(schp_frame.size())
            loss = criterionL1(warped_cloth, schp_frame)
            print("calculated loss")
            optimizer.zero_grad()
            print("zero grad")
            torch.cuda.empty_cache()
            print("empty cuda cache")
            loss.backward()
            print("did backprop")
            optimizer.step()
            print("optimizer step")

            """if (step + 1) % opt.display_count == 0:
                board_add_images(board, 'combine', visuals, step + 1)
                board.add_scalar('metric', loss.item(), step + 1)
                t = time.time() - iter_start_time
                print('step: %8d, time: %.3f, loss: %4f' % (step + 1, t, loss.item()), flush=True)"""

            if (step + 1) % opt.save_count == 0:
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
    """schp = SCHPDataset(opt)
    print("SCHP Dataset Created. Length of dataset", len(schp))
    vibe = VIBEDataset(opt)
    print("VIBE Dataset Created. Length of dataset", len(vibe))"""

    # create dataloader

    vvt_loader = torch.utils.data.DataLoader(
        vvt, batch_size=opt.batch_size, shuffle=None,
        num_workers=opt.workers)
    """schp_loader = torch.utils.data.DataLoader(
        schp, batch_size=opt.batch_size, shuffle=None,
        num_workers=opt.workers)
    vibe_loader = torch.utils.data.DataLoader(
        vibe, batch_size=opt.batch_size, shuffle=None,
        num_workers=opt.workers)"""
    '''vvt_loader = CPDataLoader(opt, vvt)
    schp_loader = CPDataLoader(opt, schp)
    vibe_loader = CPDataLoader(opt, vibe)'''

    #dataloaders = [, schp_loader, vibe_loader]
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
