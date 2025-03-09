from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import cv2
from scipy.spatial.transform import Rotation as R

from src.entities.arguments import OptimizationParams
from src.entities.losses import l1_loss
from src.entities.gaussian_model import GaussianModel
from src.entities.logger import Logger
from src.entities.datasets import BaseDataset
from src.entities.visual_odometer import VisualOdometer
from src.utils.gaussian_model_utils import build_rotation
from src.utils.tracker_utils import (compute_camera_opt_params,
                                     extrapolate_poses, multiply_quaternions,
                                     transformation_to_quaternion)
from src.utils.utils import (get_render_settings, np2torch,
                             render_gaussian_model, torch2np)
from model.GMA.network import RAFTGMA



class Motion():
    def __init__(self, config: dict, dataset: BaseDataset, logger: Logger) -> None:

        # 初始化数据集、日志和配置
        self.dataset = dataset
        self.logger = logger  
        self.config = config

        self.motion = False
        self.windows_size = 5
        self.windows = []
        self.cur_frame_id = 0


        self.motion_mask
        
        # 初始化图像转换器和优化参数
        self.transform = torchvision.transforms.ToTensor()  # 图像转换为tensor
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.flow_model = torch.nn.DataParallel(RAFTGMA)
        self.flow_model.load_state_dict(torch.load(config["flow"]["checkpoint"]))
        self.flow_model = self.flow_model.module
        self.flow_model.to('cuda')
        self.flow_model.eval()


    def detect(self):
        # Detect motion
        self.motion = True
        return self.motion

    def is_motion(self):
        return self.motion

    def reset(self):
        self.motion = False
    
    def append_windows(self):
        # index, color_data, depth_data, self.poses[index]
        frame= {}
        frame["index"],frame["image"], frame["depth"], frame["pose"]  = self.dataset[self.cur_frame_id]
        if len(self.windows) < self.windows_size:
            self.windows.append(frame)
        else:
            self.windows.pop(0)
            self.windows.append(frame)

    def update_frame(self, index):
        self.cur_frame_id = index
        self.append_windows()


    

    def get_motion_mash(self):

        flow_batch = self.windows
        
        with torch.no_grad():
            flow_img_batch = [i["image"].transpose(2,0,1) for i in flow_batch]
            img_batch = torch.stack(flow_img_batch, dim=0).to(self.device)
            cur_img = torch.from_numpy(flow_batch[-1]["image"].transpose(2,0,1))
            cur_img_batch = cur_img.repeat(img_batch.shape[0], 1, 1, 1).to(self.device)

            padder = InputPadder(img_batch.shape)  
            his_imgs, cur_imgs = padder.pad(img_batch, cur_img_batch)

            _, flow_bwd = self.flow_model(cur_imgs, his_imgs, iters=30, test_mode=True)
            flow_bwd = padder.unpad(flow_bwd).permute(0, 2, 3, 1)

            H = cur_img_batch.shape[2]
            W = cur_img_batch.shape[3]
            uv = get_uv_grid(H, W, align_corners=False)  # 获取像素坐标网格
            x1 = uv.reshape(-1, 2)  # 原始像素坐标

            # 归一化光流场到[-1,1]范围
            flow_bwd_norm = torch.stack([2.0 * flow_bwd[..., 0] / (W - 1), 
                                       2.0 * flow_bwd[..., 1] / (H - 1)], axis=-1)
            
            err_batch = []
            for i in range(flow_bwd_norm.shape[0]):
                flow_tmp = flow_bwd_norm[i].cpu()
                x2 = x1 + flow_tmp.view(-1, 2)  # 光流变换后的坐标
                
                # 使用RANSAC估计基础矩阵
                F, mask = cv2.findFundamentalMat(x1.numpy(), x2.numpy(), cv2.FM_LMEDS)
                F = torch.from_numpy(F.astype(np.float32))
                
                # 计算Sampson误差
                err = compute_sampson_error(x1, x2, F).reshape(H, W)
                fac = (H + W) / 2
                err = err * fac ** 2  # 缩放误差
                err_batch.append(err)

            
            error_batch = torch.stack(err_batch, 0)
            thresh = torch.quantile(error_batch.view(len(err_batch), -1), 0.85, dim=-1)  # 计算误差阈值
            thresh = thresh[:, None, None].repeat(1, H, W)
            err_map = torch.where(error_batch <= thresh, torch.zeros_like(error_batch), torch.ones_like(error_batch))

            finial_error_map = torch.ones_like(err_map[0])
            for j in range(err_map.shape[0]):
                finial_error_map *= err_map[j]
            
            finial_motion_map = finial_error_map.int()
            self.motion_mask = finial_motion_map


        return self.motion_mask
    
    def save_motion_mask(self):
        """
        将运动掩码转换为图像并保存
        Args:
            save_path: 保存路径，若为None则使用默认路径
        """
        if not hasattr(self, 'motion_mask') or self.motion_mask is None:
            print("运动掩码尚未生成，无法保存")
            return
        
        # 确保掩码在CPU上，并转为numpy数组
        mask = self.motion_mask.cpu().numpy()
        
        # 将二值掩码转换为可视化图像 (255表示运动区域，0表示静态区域)
        mask_img = (mask * 255).astype(np.uint8)
        
        # 可选：应用伪彩色映射以便更好地可视化
        mask_color = cv2.applyColorMap(mask_img, cv2.COLORMAP_JET)
        
        # 设置默认保存路径
        if save_path is None:
            # 使用当前帧ID作为文件名的一部分
            save_path = f"{self.cur_frame_id:06d}.png"
            
            # 如果logger有指定输出目录，则使用该目录
            if hasattr(self.logger, 'output_dir'):
                import os
                os.makedirs(os.path.join(self.logger.output_dir, "motion_masks"), exist_ok=True)
                save_path = os.path.join(self.logger.output_dir, "motion_masks", save_path)
        
        # 保存图像
        cv2.imwrite(save_path, mask_color)
        
        # 记录日志
        if hasattr(self.logger, 'log'):
            self.logger.log(f"运动掩码已保存至: {save_path}")
        else:
            print(f"运动掩码已保存至: {save_path}")
        
        return save_path



class InputPadder:
    """ Pads images such that dimensions are divisible by 8 """
    def __init__(self, dims, mode='sintel'):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // 8) + 1) * 8 - self.ht) % 8
        pad_wd = (((self.wd // 8) + 1) * 8 - self.wd) % 8
        if mode == 'sintel':
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]
        else:
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, 0, pad_ht]

    def pad(self, *inputs):
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self,x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]


def get_uv_grid(H, W, homo=False, align_corners=False, device=None):
    """
    Get uv grid renormalized from -1 to 1
    :returns (H, W, 2) tensor
    """
    if device is None:
        device = torch.device("cpu")
    yy, xx = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=device),
        torch.arange(W, dtype=torch.float32, device=device),
        indexing="ij",
    )
    if align_corners:
        xx = 2 * xx / (W - 1) - 1
        yy = 2 * yy / (H - 1) - 1
    else:
        xx = 2 * (xx + 0.5) / W - 1
        yy = 2 * (yy + 0.5) / H - 1
    if homo:
        return torch.stack([xx, yy, torch.ones_like(xx)], dim=-1)
    return torch.stack([xx, yy], dim=-1)


def compute_sampson_error(x1, x2, F):
    """
    :param x1 (*, N, 2)
    :param x2 (*, N, 2)
    :param F (*, 3, 3)
    """
    h1 = torch.cat([x1, torch.ones_like(x1[..., :1])], dim=-1)
    h2 = torch.cat([x2, torch.ones_like(x2[..., :1])], dim=-1)
    d1 = torch.matmul(h1, F.transpose(-1, -2))  # (B, N, 3)
    d2 = torch.matmul(h2, F)  # (B, N, 3)
    z = (h2 * d1).sum(dim=-1)  # (B, N)
    err = z**2 / (
        d1[..., 0] ** 2 + d1[..., 1] ** 2 + d2[..., 0] ** 2 + d2[..., 1] ** 2
    )
    return err
