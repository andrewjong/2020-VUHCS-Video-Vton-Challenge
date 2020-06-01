class VvtCompetitionDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        self.radius = 4.5
        self.opt = opt
        self.root = opt.dataroot
        self._clothes_person_dir = osp.join(self.root, "lip_clothes_person")

        self.img_h = opt.img_h
        self.img_w = opt.img_w

        if opt.isTrain:
            self._keypoints_dir = osp.join(self.root, "lip_train_frames_keypoint")
            self._frames_dir = osp.join(self.root, "lip_train_frames")
            raise NotImplementedError(
                "Train is not yet implemented for vvt competition")
        else:
            self._keypoints_dir = osp.join(self.root, "lip_test_frames_keypoint")
            self._frames_dir = osp.join(self.root, "lip_test_frames")

        self.keypoints = glob(f"{self._keypoints_dir}/**/*.json")
        assert len(self.keypoints) > 0

        self.to_tensor = transforms.Compose([transforms.ToTensor()])

    def randomFlip(self, x, x_target, pose, pose_target, mask, mask_target):
        # random horizontal flip
        rand = random.randint(0, 1)
        if (rand == 1):
            x = np.flip(x, axis=1).copy()
            x_target = np.flip(x_target, axis=1).copy()
            pose = np.flip(pose, axis=1).copy()
            pose_target = np.flip(pose_target, axis=1).copy()
            mask = np.flip(mask, axis=1).copy()
            mask_target = np.flip(mask_target, axis=1).copy()
        return x, x_target, pose, pose_target, mask, mask_target

    def get_target_frame(self, index):
        """ Gets the target frame that corresponds to the keypoint at this index """
        _pose_name = self.keypoints[index]
        _pose_name = _pose_name.replace("_keypoints.json", ".png")
        just_folder_and_file = _pose_name.split("/")[-2:]
        frame_path = osp.join(self._frames_dir, *just_folder_and_file)
        frame = self.to_tensor(Image.open(frame_path))
        frame = self._pad_width_up(frame)
        return frame

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

        # draw a circle around the joint on the appropriate channel
        for i in range(point_num):
            pointx = pose_data[i, 0]
            pointy = pose_data[i, 1]
            if pointx > 1 and pointy > 1:
                rr, cc = draw.circle(pointy, pointx, self.radius, shape=(self.img_h, self.img_w))
                pose_map[i, rr, cc] = 1

        # add padding to the w/ h/
        pose_map = self._pad_width_up(pose_map, value=-1)# make the image 256x256
        assert all(i == -1 or i == 1 for i in torch.unique(pose_map)), f"{torch.unique(pose_map)}"
        return pose_map

    def _get_input_person_path_from_index(self, index):
        """ Returns the path to the person image file that is used as input """
        pose_name = self.keypoints[index]
        person_id = pose_name.split("/")[-2]
        folder = osp.join(self._clothes_person_dir, person_id)

        files = os.listdir(folder)
        person_image_name = [f for f in files if f.endswith(".png")][0]
        assert person_image_name.endswith(".png"), f"person images should have .png extensions: {person_image_name}"
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
        person_tensor = self.to_tensor(person_image)
        person_tensor = self._pad_width_up(person_tensor)
        return person_tensor


    def __getitem__(self, index):
        image = self.get_input_person(index)  # (3, 256, 256)
        try:
            pose_target = self.get_input_person_pose(index, target_width=256)  # (18, 256, 256)
        except IndexError as e:
            print(e.__traceback__)
            print(f"[WARNING]: no pose found {self.keypoints[index]}")
            pose_target = torch.zeros(18, 256, 256)
        image_target = self.get_target_frame(index)

        assert image.shape[-2:] == pose_target.shape[-2:], f"hxw don't match: image {image.shape}, pose {pose_target.shape}"
        assert image.shape[-2:] == image_target.shape[-2:], f"hxw don't match: image {image.shape}, pose {pose_target.shape}"

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
        guide = pose_target
        target = (image_target / 255) * 2 - 1
        # Put data into [input, guide, target]
        return {'A': input, "guide_path": self.keypoints[index], 'guide': guide, 'B': target}

    def __len__(self):
        return len(self.keypoints)

    def name(self):
        return 'VvtCompetitionDataset'