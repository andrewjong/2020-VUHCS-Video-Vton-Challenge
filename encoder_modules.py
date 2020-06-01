import torch
from torch import nn

class Encoder(nn.Module):
    def __init__(self, input_size, num_fmaps, output_channels=3):
        super(Encoder, self).__init__()

        def encoder_block(in_feat, out_feat):
            block = [
                nn.Linear(in_feat,out_feat),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(out_feat),  # check out 2nd parameter information
            ]

            return block

        self.model = nn.Sequential(
            # upsample input vector
            *encoder_block(input_size, 5000),
            *encoder_block(5000, 5000),
            *encoder_block(5000, num_fmaps),
            #*encoder_block(num_fmaps, num_fmaps),
            # upsample input feature maps till you get desired feature maps
            #*encoder_block(num_fmaps * 8, num_fmaps * 4),
            #*encoder_block(num_fmaps * 4, num_fmaps * 2),
            #*encoder_block(num_fmaps * 2, num_fmaps),
            # go from feature maps to generated images
            #nn.ConvTranspose2d(num_fmaps, output_channels, 4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, img):
        # TODO
        #z = torch.randn(1, z, 1, 1)
        img = self.model(img)
        return img