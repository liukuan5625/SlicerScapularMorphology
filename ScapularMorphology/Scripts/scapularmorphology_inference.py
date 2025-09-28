import argparse
import os
import nibabel as nib
import numpy as np
import torch
import warnings
import scipy.ndimage as ndi
from monai import transforms
from segmentation_models_pytorch import Unet
from monai.networks.nets import SegResNetDS
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description="Scapula inference pipeline")
parser.add_argument("--scapula_dir", type=str, help="scapula data directory")
parser.add_argument("--humerus_dir", type=str, help="humerus data directory")
parser.add_argument("--output_dir_2d", default="./glenoid.nii.gz", type=str, help="glenoid pred output directory")
parser.add_argument("--output_dir_3d", default="./landmark.csv", type=str, help="landmarks pred output directory")

parser.add_argument("--weight_dir_2d", default="./", type=str, help="weight checkpoint directory")
parser.add_argument("--weight_model_name_2d", default="model_2d.pt", type=str, help="weight model name")
parser.add_argument("--in_channels_2d", default=9, type=int, help="number of input channels")
parser.add_argument("--out_channels_2d", default=2, type=int, help="number of output channels")
parser.add_argument("--size_2d", default=512, type=int, help="ct resolution")
parser.add_argument("--scapula_humerus_dist", default=25, type=int, help="the distance between scapula and humerus for glenoid extraction")

parser.add_argument("--weight_dir_3d", default="./", type=str, help="weight checkpoint directory")
parser.add_argument("--weight_model_name_3d", default="model_3d.pt", type=str, help="weight model name")
parser.add_argument("--in_channels_3d", default=1, type=int, help="number of input channels")
parser.add_argument("--out_channels_3d", default=5, type=int, help="number of output channels")
parser.add_argument("--size_x_3d", default=128, type=int, help="3D ct resolution")
parser.add_argument("--size_y_3d", default=128, type=int, help="3D ct resolution")
parser.add_argument("--size_z_3d", default=64, type=int, help="3D ct resolution")
parser.add_argument("--spacing_x_3d", default=2, type=int, help="3D ct spacing")
parser.add_argument("--spacing_y_3d", default=2, type=int, help="3D ct spacing")
parser.add_argument("--spacing_z_3d", default=4, type=int, help="3D ct spacing")

parser.add_argument("--minISrange", type=float, help="minimum IS range")
parser.add_argument("--maxISrange", type=float, help="maximum IS range")

parser.add_argument("--heat", default=False, type=bool, help="xray and ct resolution")
parser.add_argument("--cpu", default=False, type=bool, help="xray and ct resolution")


def main():
    args = parser.parse_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')

    scapula_img = transforms.LoadImage()(args.scapula_dir)
    humerus_img = transforms.LoadImage()(args.humerus_dir)
    # GLENOID INFER
    glenoid_img = cal_glenoid_img(scapula_img, humerus_img, args)
    glenoid_center_idx, glenoid_center_loc = cal_glenoid_seg(glenoid_img, args)
    # LANDMARKS INFER
    cal_landmarks(scapula_img, glenoid_center_idx, glenoid_center_loc, args)


def cal_glenoid_img(sca_img, hum_img, args):
    ori_aff_mat = sca_img.meta['original_affine']
    xy_spacing = abs(np.sum(ori_aff_mat[:, 0]))
    z_spacing = np.sum(ori_aff_mat[:, 2])
    z_origin = ori_aff_mat[2, 3]
    z_slices = sca_img.shape[-1]

    sca_img = sca_img.permute(2, 0, 1)
    hum_img = hum_img.permute(2, 0, 1)
    gle_idx = []
    if args.minISrange is None or args.maxISrange is None:
        for i, sh in enumerate(zip(sca_img, hum_img)):
            s, h = sh[0], sh[1]
            if np.sum(s) != 0 and np.sum(h) != 0:
                d_s = ndi.distance_transform_edt(1 - s)
                h_p = np.argwhere(h == 1)
                distances = d_s[tuple(h_p.T)]
                min_distance = distances.min() * xy_spacing
                if min_distance < args.scapula_humerus_dist:
                    gle_idx.append(i)
    else:
        min_idx = abs(int(np.round((args.minISrange - z_origin) / z_spacing)))
        max_idx = abs(int(np.round((args.maxISrange - z_origin) / z_spacing)))
        gle_idx.append(min(min_idx, max_idx))
        gle_idx.append(max(min_idx, max_idx))

    gle_idx = list(range(max(gle_idx[0] - args.in_channels_2d, 0), min(gle_idx[-1] + args.in_channels_2d, z_slices)))
    if len(gle_idx) == 0:
        assert 1 == 0

    gle_img = sca_img.permute(1, 2, 0)[:, :, gle_idx]
    translate = ori_aff_mat[:3, :3] @ np.array([0, 0, gle_idx[0]])
    affine = ori_aff_mat
    affine[:-1, -1] = affine[:-1, -1] + translate
    gle_img.meta['affine'] = affine
    return gle_img


def cal_glenoid_seg(gle_img, args):
    infer_transform = transforms.Compose([
        transforms.CenterSpatialCrop(roi_size=[args.size_2d, args.size_2d]),
        transforms.ThresholdIntensity(threshold=0.5, above=False, cval=1)
    ])

    model = Unet(
        encoder_name="resnet34",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
        in_channels=args.in_channels_2d,
        classes=args.out_channels_2d,  # model output channels (number of classes in your dataset)
    )
    weight_dir = args.weight_dir_2d
    model_name = args.weight_model_name_2d
    weight_pth = os.path.join(weight_dir, model_name)
    model_dict = torch.load(weight_pth, weights_only=False, map_location=torch.device(args.device))["state_dict"]
    model.load_state_dict(model_dict, strict=True)
    model.to(args.device)
    model.eval()
    # IMPORTANT
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.train(True)

    gle_img = gle_img.permute(2, 0, 1)
    slice_num = len(gle_img)
    slice_interval = args.in_channels_2d // 2
    slice_idx = range(slice_interval, slice_num - slice_interval)

    gle_img = infer_transform(gle_img).to(args.device)
    gle_out = np.zeros_like(gle_img)
    with torch.no_grad():
        for idx in slice_idx:
            torch.cuda.empty_cache()
            infer_input = gle_img[idx-slice_interval:idx+slice_interval+1]
            infer_output = model(infer_input.unsqueeze(0)).cpu()
            infer_output = np.argmax(infer_output.numpy(), axis=1, keepdims=False)
            gle_out[idx:idx + 1] = infer_output[0]

    aff_mat = gle_img.meta['affine']
    gle_out = gle_out.transpose(1, 2, 0)

    # largest connected component
    connect = ndi.generate_binary_structure(3, 2)
    connect_arrays, num = ndi.label(gle_out, connect)
    connect_nums = ndi.sum(gle_out, connect_arrays, range(1, num + 1))
    max_connect_num = np.argmax(connect_nums) + 1
    gle_lcc = (connect_arrays == max_connect_num)
    nib.save(
        nib.Nifti1Image(gle_img.astype(np.uint8), affine=aff_mat),
        args.output_dir_2d.replace('.nii.gz', '_input.nii.gz')
    )
    nib.save(
        nib.Nifti1Image(gle_lcc.astype(np.uint8), affine=aff_mat),
        args.output_dir_2d
    )

    gc_idx = np.mean(np.nonzero(gle_lcc), 1)
    gc_loc = np.float32(np.matmul(aff_mat, np.concatenate([gc_idx, [1]]))[:-1])
    gc_idx = tuple(np.int8(np.round(gc_idx)))
    return gc_idx, gc_loc


def cal_landmarks(sca_img, gle_idx, gle_loc, args):
    infer_transform = transforms.Compose([
        transforms.EnsureChannelFirst(channel_dim='no_channel'),
        transforms.Spacing(pixdim=[args.spacing_x_3d, args.spacing_y_3d, args.spacing_z_3d], mode='nearest'),
        transforms.CenterSpatialCrop(roi_size=[args.size_x_3d, args.size_y_3d, args.size_z_3d]),
        transforms.SpatialPad(spatial_size=[args.size_x_3d, args.size_y_3d, args.size_z_3d]),
    ])

    model = SegResNetDS(init_filters=32,
                        blocks_down=(1, 2, 2, 4, 4),
                        norm='INSTANCE_NVFUSER',
                        in_channels=args.in_channels_3d,
                        out_channels=args.out_channels_3d,
                        dsdepth=4)
    weight_dir = args.weight_dir_3d
    model_name = args.weight_model_name_3d
    weight_pth = os.path.join(weight_dir, model_name)
    model_dict = torch.load(weight_pth, weights_only=False, map_location=torch.device(args.device))["state_dict"]
    model.load_state_dict(model_dict, strict=True)
    model = torch.nn.Sequential(model, torch.nn.Sigmoid())
    model.to(args.device)
    model.eval()

    landmarks_idx = [gle_idx]
    landmarks_loc = [gle_loc]
    infer_input = infer_transform(sca_img).to(args.device)
    aff_mat = infer_input.meta['affine']
    with torch.no_grad():
        output = torch.nn.Sigmoid()(model(infer_input.unsqueeze(0))[0]).cpu().numpy()

    sigma = (2, 2, 4)
    landmarks_name = np.array(['GC', 'TS', 'AI', 'PC', 'AC', 'AA'], dtype=np.str_).reshape(-1, 1)
    for i, o in enumerate(output):
        o = ndi.gaussian_filter(np.float32(o), sigma)
        o_idx = np.mean(np.argwhere(o > 0.95 * np.max(o)), 0)
        o_loc = np.matmul(aff_mat, np.concatenate([o_idx, [1]]))
        landmarks_idx.append(tuple(np.int8(np.round(o_idx))))
        landmarks_loc.append(np.float32(o_loc[:-1]))

        if args.heat:
            nib.save(nib.Nifti1Image(o.astype(np.float32), affine=aff_mat),
                     args.output_dir_3d.replace('.csv', '_'+landmarks_name[i+1, 0]+'.nii.gz'))

    space_header = infer_input.meta['space']
    space_header = 'label,' + space_header[0].lower() + ',' + space_header[1].lower() + ',' + space_header[2].lower()
    np.savetxt(args.output_dir_3d, np.concatenate([landmarks_name, landmarks_loc], 1),
               fmt="%s", delimiter=',', header=space_header, comments='')


main()

