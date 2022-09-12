import os
import argparse
import torch
from torch.utils.data import DataLoader
import numpy as np
import json
import pickle

from models import networks
from models import ops
from dataset.static_pose_shape_final import OneStyleShapeHF
import global_var
from trainer import base_trainer

device = torch.device("cuda:0")
# device = torch.device("cpu")

import geometricUtils as lossUtils
import wandb


class HFTrainer(base_trainer.Trainer):
    """Implements trainer class for TailorNet high frequency predictor.

    It overloads some functions of base_trainer.Trainer class.
    """

    def load_dataset(self, split):
        params = self.params
        shape_idx, style_idx = params['shape_style'].split('_')

        dataset = OneStyleShapeHF(self.garment_class, shape_idx=shape_idx, style_idx=style_idx, split=split,
                                  gender=self.gender, smooth_level=params['smooth_level'])
        shuffle = True if split == 'train' else False
        if split == 'train' and len(dataset) > params['batch_size']:
            drop_last = True
        else:
            drop_last = False
        dataloader = DataLoader(dataset, batch_size=self.bs, num_workers=0, shuffle=shuffle,
                                drop_last=drop_last)
        return dataset, dataloader

    def build_model(self):
        params = self.params
        model = getattr(networks, self.model_name)(
            input_size=72, output_size=self.vert_indices.shape[0] * 3,
            num_layers=params['num_layers'],
            hidden_size=params['hidden_size'])
        return model

    def one_step(self, inputs):
        gt_verts, smooth_verts, thetas, _, _, _ = inputs

        thetas = ops.mask_thetas(thetas, self.garment_class)
        gt_verts = gt_verts.to(device)
        smooth_verts = smooth_verts.to(device)
        thetas = thetas.to(device)

        # predicts residual over smooth groundtruth.
        pred_verts = self.model(thetas).view(gt_verts.shape) + smooth_verts

        # L1 loss
        data_loss = (pred_verts - gt_verts).abs().sum(-1).mean() + \
                    self.patch_loss(pred_verts, gt_verts, torch.from_numpy(self.f),
                                    patch_ratio=1, p=1, weight=1)
        wandb.log({'train_loss': data_loss})
        return pred_verts, data_loss

    def update_metrics(self, metrics, inputs, outputs):
        gt_verts = inputs[0]
        pred_verts = outputs
        dist = ops.verts_dist(gt_verts, pred_verts.cpu()) * 1000.
        metrics['val_dist'].update(dist.item(), gt_verts.shape[0])

    def visualize_batch(self, inputs, outputs, epoch):
        gt_verts, smooth_verts, thetas, betas, gammas, idxs = inputs
        new_inputs = (gt_verts, thetas, betas, gammas, idxs)
        super(HFTrainer, self).visualize_batch(new_inputs, outputs, epoch)

    def patch_loss(self, v, v_gt, f, patch_ratio=1, p=1, weight=1):
        # v = v / 100
        # v_gt = v_gt / 100
        # we will use the given normals to reduce some calculation time.
        vn = lossUtils.get_vertex_normals(v, f.unsqueeze(0).repeat(v.shape[0], 1, 1))
        vn_gt = lossUtils.get_vertex_normals(v_gt, f.unsqueeze(0).repeat(v.shape[0], 1, 1))
        # vn_gt = lossUtils.get_vertex_normals(v_gt, f)

        # we don't need the faces because it is given as partial_f in the patch dictionary.
        # f = f[0]
        for patches, patch_menu in zip(self.patches_list, self.patch_menu_list):
            indices = np.random.choice(len(patches), int(len(patches) * patch_ratio))
            loss = 0
            for idx in indices:
                patch = patches[idx]
                v_ref = patch_menu[idx]

                v_patch = patch['v_patch']

                partial_f = torch.from_numpy(patch['partial_f']).to(device)
                partial_e = torch.from_numpy(np.array(patch['partial_e'])).long().to(device).detach()
                # xy = torch.from_numpy(patch['xy']).to(self.dev)
                # xy_shape = patch['xy_shape']
                # face_indices = torch.from_numpy(patch['face_indices']).to(self.dev)

                # here we will use the same normal directions for both tangent plane to reduce loss from orientation
                # differences, here we only care about the local texture loss.
                tangent_plane_normal = vn_gt[:, v_ref]
                tangent_plane_normal_gt = vn_gt[:, v_ref]
                v_vec = v[:, v_patch] - v[:, v_ref].unsqueeze(-2).repeat(1, len(v_patch), 1)
                v_vec_gt = v_gt[:, v_patch] - v_gt[:, v_ref].unsqueeze(-2).repeat(1, len(v_patch), 1)
                dist = v_vec @ tangent_plane_normal.unsqueeze(-1)
                dist_gt = v_vec_gt @ tangent_plane_normal_gt.unsqueeze(-1)
                # partial_v = torch.concat([self.v_uv[v_uv_patch].unsqueeze(0).repeat(dist.shape[0], 1, 1), dist], dim=-1)
                # partial_v_gt = torch.concat([self.v_uv[v_uv_patch].unsqueeze(0).repeat(dist.shape[0], 1, 1), dist_gt], dim=-1)

                # triangles_uv = self.v_uv[v_uv_patch][partial_f]

                # interpolated_vals = lossUtils.interpolate_points_in_batch(triangles_uv, partial_f, face_indices, xy, dist)
                # interpolated_vals_gt = lossUtils.interpolate_points_in_batch(triangles_uv, partial_f, face_indices, xy, dist_gt)

                tv_gt = lossUtils.calc_total_variation_for_each_point(vn_gt[:, v_patch], partial_e, len(v_patch),
                                                                      aug_factor=1)
                tv = lossUtils.calc_total_variation_for_each_point(vn[:, v_patch], partial_e, len(v_patch),
                                                                   aug_factor=1)
                loss += torch.mean(((dist.squeeze(-1) - dist_gt.squeeze(-1)).abs() ** p) * tv)
                # print("patch L2 weighted loss: ", loss)
                if weight > 0:
                    # print("tv loss: ", torch.mean(torch.square(tv - tv_gt)))
                    loss += torch.mean(torch.square(tv - tv_gt))

        return loss


class Runner(object):
    """A helper class to load a trained model."""
    def __init__(self, ckpt, params):
        model_name = params['model_name']
        garment_class = params['garment_class']

        with open(os.path.join(global_var.DATA_DIR, global_var.GAR_INFO_FILE), 'rb') as f:
            class_info = pickle.load(f)
        output_size = len(class_info[garment_class]['vert_indices']) * 3

        self.model = getattr(networks, model_name)(
            input_size=72, output_size=output_size,
            hidden_size=params['hidden_size'] if 'hidden_size' in params else 1024,
            num_layers=params['num_layers'] if 'num_layers' in params else 3
        )
        self.garment_class = params['garment_class']

        print("loading {}".format(ckpt))
        if torch.cuda.is_available():
            self.model.cuda()
            state_dict = torch.load(ckpt)
        else:
            state_dict = torch.load(ckpt,map_location='cpu')
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def forward(self, thetas, betas=None, gammas=None):
        thetas = ops.mask_thetas(thetas=thetas, garment_class=self.garment_class)
        pred_verts = self.model(thetas)
        return pred_verts

    def cuda(self):
        self.model.cuda()

    def to(self, device):
        self.model.to(device)


def get_best_runner(log_dir, epoch_num=None):
    """Returns a trained model runner given the log_dir."""
    ckpt_dir = log_dir
    with open(os.path.join(ckpt_dir, 'params.json')) as jf:
        params = json.load(jf)

    # if epoch_num is not given then pick up the best epoch
    if epoch_num is None:
        ckpt_path = os.path.join(ckpt_dir, 'lin.pth.tar')
    else:
        # with open(os.path.join(ckpt_dir, 'best_epoch')) as f:
        #     best_epoch = int(f.read().strip())
        best_epoch = epoch_num
        ckpt_path = os.path.join(ckpt_dir, "{:04d}".format(best_epoch), 'lin.pth.tar')

    runner = Runner(ckpt_path, params)
    return runner


def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_config', default='')

    parser.add_argument('--garment_class', default="old-t-shirt")
    parser.add_argument('--gender', default="female")
    parser.add_argument('--shape_style', nargs='+')

    # some training hyper parameters
    parser.add_argument('--vis_freq', default=16, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--weight_decay', default=1e-6, type=float)
    parser.add_argument('--max_epoch', default=800, type=int)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--checkpoint', default="")

    # name under which experiment will be logged
    parser.add_argument('--log_name', default="tn_hf")

    # smooth_level=1 will train HF for that smoothness level
    parser.add_argument('--smooth_level', default=1, type=int)

    # model specification.
    parser.add_argument('--model_name', default="FullyConnected")
    parser.add_argument('--num_layers', default=3)
    parser.add_argument('--hidden_size', default=1024)

    # small experiment description
    parser.add_argument('--note', default="TailorNet high frequency prediction")

    # patch loss params
    parser.add_argument('--patch_dir_list', default=['..\\Patch_info\\patches_200_with_uv',
                                                     '..\\Patch_info\\patches_400_with_uv',
                                                     '..\\Patch_info\\patches_800_with_uv'])

    args = parser.parse_args()
    params = args.__dict__

    # load params from local config if provided
    if os.path.exists(params['local_config']):
        print("loading config from {}".format(params['local_config']))
        with open(params['local_config']) as f:
            lc = json.load(f)
        for k, v in lc.items():
            params[k] = v
    return params


def main():
    params = parse_argument()
    # shape_styles = params['shape_style']
    shape_styles = ['000_000']

    for ss in shape_styles:
        params['shape_style'] = ss
        print("start training {} on {}".format(params['garment_class'], ss))
        trainer = HFTrainer(params)

        wandb.init(
            project='tailornet_with_patches',
            name='lr=%.4f' % params['lr']
        )
        wandb.watch(trainer.model)

        for i in range(params['start_epoch'], params['max_epoch']):
            print("epoch: {}".format(i))
            trainer.train(i)
            if i % 20 == 0:
                trainer.validate(i)
            # if i % 40 == 0:
            #     trainer.save_ckpt(i)

        trainer.save_ckpt(params['max_epoch']-1)
        trainer.write_log()
        print("safely quit!")

        break


if __name__ == '__main__':
    main()