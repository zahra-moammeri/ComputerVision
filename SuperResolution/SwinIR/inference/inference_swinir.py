# Modified from https://github.com/JingyunLiang/SwinIR
import argparse
import cv2
import glob
import numpy as np
import os
import torch
from torch.nn import functional as F

from swinir.archs.swinir_arch import SwinIR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='datasets/Set5/LR_bicubic/X2', help='input test image folder')
    parser.add_argument('--output', type=str, default='results/SwinIR/Set5', help='output folder')
    parser.add_argument('--task', type=str, default='classical_sr', help='classical_sr, lightweight_sr, real_sr')
    # TODO: it now only supports sr
    parser.add_argument('--patch_size', type=int, default=48, help='training patch size')
    parser.add_argument('--scale', type=int, default=2, help='scale factor: 2') 
    parser.add_argument('--model_path', type=str, default='experiments/pretrained_models/net_g_latest.pth')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # set up model
    model = define_model(args)
    model.eval()
    model = model.to(device)

    for idx, path in enumerate(sorted(glob.glob(os.path.join(args.input, '*')))):
        # read image
        imgname = os.path.splitext(os.path.basename(path))[0]
        print('Testing', idx, imgname)
        # read image
        img = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
        img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
        img = img.unsqueeze(0).to(device)

        # inference
        with torch.no_grad(): 

            window_size = 8
            # pad input image to be a multiple of window_size
            mod_pad_h, mod_pad_w = 0, 0
            _, _, h, w = img.size()
            if h % window_size != 0:
                mod_pad_h = window_size - h % window_size
            if w % window_size != 0:
                mod_pad_w = window_size - w % window_size
            img = F.pad(img, (0, mod_pad_w, 0, mod_pad_h), 'reflect')

            output = model(img)
            _, _, h, w = output.size()
            output = output[:, :, 0:h - mod_pad_h * args.scale, 0:w - mod_pad_w * args.scale]

        # save image
        output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        if output.ndim == 3:
            output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
        output = (output * 255.0).round().astype(np.uint8)
        cv2.imwrite(os.path.join(args.output, f'{imgname}_SwinIR.png'), output)


def define_model(args):
    # 001 classical image sr
    if args.task == 'classical_sr':
        model = SwinIR(
            upscale=args.scale,
            in_chans=3,
            img_size=args.patch_size,
            window_size=8,
            img_range=1.,
            depths=[6, 6, 6, 6, 6, 6],
            embed_dim=180,
            num_heads=[6, 6, 6, 6, 6, 6],
            mlp_ratio=2,
            upsampler='pixelshuffle',
            resi_connection='1conv')

    # 002 lightweight image sr
    # use 'pixelshuffledirect' to save parameters
    elif args.task == 'lightweight_sr':
        model = SwinIR(
            upscale=args.scale,
            in_chans=3,
            img_size=64,
            window_size=8,
            img_range=1.,
            depths=[6, 6, 6, 6],
            embed_dim=60,
            num_heads=[6, 6, 6, 6],
            mlp_ratio=2,
            upsampler='pixelshuffledirect',
            resi_connection='1conv')

    # 003 real-world image sr
    elif args.task == 'real_sr':
        if not args.large_model:
            # use 'nearest+conv' to avoid block artifacts
            model = SwinIR(
                upscale=4,
                in_chans=3,
                img_size=64,
                window_size=8,
                img_range=1.,
                depths=[6, 6, 6, 6, 6, 6],
                embed_dim=180,
                num_heads=[6, 6, 6, 6, 6, 6],
                mlp_ratio=2,
                upsampler='nearest+conv',
                resi_connection='1conv')
        else:
            # larger model size; use '3conv' to save parameters and memory; use ema for GAN training
            model = SwinIR(
                upscale=4,
                in_chans=3,
                img_size=64,
                window_size=8,
                img_range=1.,
                depths=[6, 6, 6, 6, 6, 6, 6, 6, 6],
                embed_dim=248,
                num_heads=[8, 8, 8, 8, 8, 8, 8, 8, 8],
                mlp_ratio=2,
                upsampler='nearest+conv',
                resi_connection='3conv')


    loadnet = torch.load(args.model_path)
    if 'params_ema' in loadnet:
        keyname = 'params_ema'
    else:
        keyname = 'params'
    model.load_state_dict(loadnet[keyname], strict=True)

    return model


if __name__ == '__main__':
    main()
