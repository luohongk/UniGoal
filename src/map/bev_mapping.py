import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import cv2

from src.utils.fmm.depth_utils import (
    get_point_cloud_from_z_t,
    transform_camera_view_t,
    transform_pose_t,
    splat_feat_nd
)
from src.utils.model import get_grid, ChannelPool
from src.utils.camera import get_camera_matrix


class Mapping(nn.Module):
    def __init__(self, args):
        super(Mapping, self).__init__()

        self.device = args.device
        self.screen_h = args.frame_height
        self.screen_w = args.frame_width
        self.resolution = args.map_resolution
        self.z_resolution = args.map_resolution
        self.map_size_cm = args.map_size_cm // args.global_downscaling
        self.n_channels = 3
        self.vision_range = args.vision_range
        self.dropout = 0.5
        self.fov = args.hfov
        self.du_scale = args.du_scale
        self.cat_pred_threshold = args.cat_pred_threshold
        self.exp_pred_threshold = args.exp_pred_threshold
        self.map_pred_threshold = args.map_pred_threshold
        self.num_sem_categories = args.num_sem_categories

        self.max_height = int(360 / self.z_resolution)
        self.min_height = int(-80 / self.z_resolution)
        self.agent_height = args.camera_height * 100.
        self.shift_loc = [self.vision_range *
                          self.resolution // 2, 0, np.pi / 2.0]
        self.camera_matrix = get_camera_matrix(
            self.screen_w, self.screen_h, self.fov)
        self.vfov = np.arctan((self.screen_h/2.) / self.camera_matrix.f)
        self.min_vision = self.agent_height / np.tan(self.vfov) # cm

        self.pool = ChannelPool(1)

        vr = self.vision_range

        self.init_grid = torch.zeros(
            args.num_processes, 1 + self.num_sem_categories, vr, vr,
            self.max_height - self.min_height
        ).float().to(self.device)
        self.feat = torch.ones(
            args.num_processes, 1 + self.num_sem_categories,
            self.screen_h // self.du_scale * self.screen_w // self.du_scale
        ).float().to(self.device)

    def forward(self, obs, pose_obs, maps_last, poses_last, agent_heights):
        bs, c, h, w = obs.size()
        depth = obs[:, 3, :, :]

        point_cloud_t = get_point_cloud_from_z_t(
            depth, self.camera_matrix, self.device, scale=self.du_scale)

        agent_view_t = transform_camera_view_t(
            point_cloud_t, agent_heights * 100., 0, self.device)

        agent_view_centered_t = transform_pose_t(
            agent_view_t, self.shift_loc, self.device)

        max_h = self.max_height
        min_h = self.min_height
        xy_resolution = self.resolution
        z_resolution = self.z_resolution
        vision_range = self.vision_range
        XYZ_cm_std = agent_view_centered_t.float()
        XYZ_cm_std[..., :2] = (XYZ_cm_std[..., :2] / xy_resolution)
        XYZ_cm_std[..., :2] = (XYZ_cm_std[..., :2] -
                               vision_range // 2.) / vision_range * 2.
        XYZ_cm_std[..., 2] = XYZ_cm_std[..., 2] / z_resolution
        XYZ_cm_std[..., 2] = (XYZ_cm_std[..., 2] -
                              (max_h + min_h) // 2.) / (max_h - min_h) * 2.
        self.feat[:, 1:, :] = nn.AvgPool2d(self.du_scale)(
            obs[:, 4:, :, :]
        ).view(bs, c - 4, h // self.du_scale * w // self.du_scale)

        XYZ_cm_std = XYZ_cm_std.permute(0, 3, 1, 2)
        XYZ_cm_std = XYZ_cm_std.view(XYZ_cm_std.shape[0],
                                     XYZ_cm_std.shape[1],
                                     XYZ_cm_std.shape[2] * XYZ_cm_std.shape[3])

        voxels = splat_feat_nd(
            self.init_grid * 0., self.feat, XYZ_cm_std).transpose(2, 3)

        min_z = int(35 / z_resolution - min_h)
        max_z = int((self.agent_height + 1) / z_resolution - min_h)
        floor_z = int(-35 / z_resolution - min_h)

        agent_height_proj = voxels[..., min_z:max_z].sum(4)
        all_height_proj = voxels.sum(4)


        min_vision_std = int(self.min_vision // z_resolution)
        around_floor_proj = voxels[..., :min_z].sum(4) 
        under_floor_proj = voxels[..., floor_z:min_z].sum(4) 
        under_floor_proj = (under_floor_proj == 0.0).float()# have floor = 0, no floor or not deteced = 1
        under_floor_proj = under_floor_proj * around_floor_proj # no floor and detected = 1

        replace_element = torch.ones_like(depth[:, -1, :]) * self.min_vision
        re_depth = torch.where(depth[:, -1, :] < 3000, depth[:, -1, :], replace_element)
        count = ((re_depth - self.min_vision - 60) > 0).sum(dim=1)
        index = torch.nonzero(count > (re_depth.shape[1] / 4))

        under_floor_proj[index, 0:1, min_vision_std:min_vision_std+1, \
                (self.vision_range-6)//2 : (self.vision_range+6)//2] \
                    = 1.

        fp_map_pred = agent_height_proj[:, 0:1, :, :] + under_floor_proj[:, 0:1, :, :]
        fp_exp_pred = all_height_proj[:, 0:1, :, :]
        fp_map_pred = fp_map_pred / self.map_pred_threshold
        fp_exp_pred = fp_exp_pred / self.exp_pred_threshold
        fp_map_pred = torch.clamp(fp_map_pred, min=0.0, max=1.0)
        fp_exp_pred = torch.clamp(fp_exp_pred, min=0.0, max=1.0)

        pose_pred = poses_last

        agent_view = torch.zeros(bs, c,
                                 self.map_size_cm // self.resolution,
                                 self.map_size_cm // self.resolution
                                 ).to(self.device)

        x1 = self.map_size_cm // (self.resolution * 2) - self.vision_range // 2
        x2 = x1 + self.vision_range
        y1 = self.map_size_cm // (self.resolution * 2)
        y2 = y1 + self.vision_range
        agent_view[:, 0:1, y1:y2, x1:x2] = fp_map_pred
        agent_view[:, 1:2, y1:y2, x1:x2] = fp_exp_pred
        agent_view[:, 4:, y1:y2, x1:x2] = torch.clamp(
            agent_height_proj[:, 1:, :, :] / self.cat_pred_threshold,
            min=0.0, max=1.0)

        corrected_pose = pose_obs

        def get_new_pose_batch(pose, rel_pose_change):

            pose[:, 1] += rel_pose_change[:, 0] * \
                torch.sin(pose[:, 2] / 57.29577951308232) \
                + rel_pose_change[:, 1] * \
                torch.cos(pose[:, 2] / 57.29577951308232)
            pose[:, 0] += rel_pose_change[:, 0] * \
                torch.cos(pose[:, 2] / 57.29577951308232) \
                - rel_pose_change[:, 1] * \
                torch.sin(pose[:, 2] / 57.29577951308232)
            pose[:, 2] += rel_pose_change[:, 2] * 57.29577951308232

            pose[:, 2] = torch.fmod(pose[:, 2] - 180.0, 360.0) + 180.0
            pose[:, 2] = torch.fmod(pose[:, 2] + 180.0, 360.0) - 180.0

            return pose

        current_poses = get_new_pose_batch(poses_last, corrected_pose)
        st_pose = current_poses.clone().detach()

        st_pose[:, :2] = - (st_pose[:, :2]
                            * 100.0 / self.resolution
                            - self.map_size_cm // (self.resolution * 2)) /\
            (self.map_size_cm // (self.resolution * 2))
        st_pose[:, 2] = 90. - (st_pose[:, 2])

        rot_mat, trans_mat = get_grid(st_pose, agent_view.size(),
                                      self.device)

        rotated = F.grid_sample(agent_view, rot_mat, align_corners=True)
        translated = F.grid_sample(rotated, trans_mat, align_corners=True)

        maps2 = torch.cat((maps_last.unsqueeze(1), translated.unsqueeze(1)), 1)

        map_pred, _ = torch.max(maps2, 1)

        return fp_map_pred, map_pred, pose_pred, current_poses



class BEV_Map():
    def __init__(self, args):
        self.args = args

        self.num_scenes = self.args.num_processes
        nc = self.args.num_sem_categories + 4  # num channels
        self.device = self.args.device

        self.map_size = self.args.map_size
        self.global_width = self.args.global_width
        self.global_height = self.args.global_height
        self.local_width = self.args.local_width
        self.local_height = self.args.local_height

        self.mapping_module = Mapping(self.args).to(self.device)
        self.mapping_module.eval()

        # Initializing full and local map
        self.full_map = torch.zeros(self.num_scenes, nc, self.global_width, self.global_height).float().to(self.device)
        self.local_map = torch.zeros(self.num_scenes, nc, self.local_width,
                                self.local_height).float().to(self.device)
        
        # 初始化价值地图 - 全局和局部
        self.full_value_map = torch.zeros(self.num_scenes, 1, self.global_width, self.global_height).float().to(self.device)
        self.local_value_map = torch.zeros(self.num_scenes, 1, self.local_width, self.local_height).float().to(self.device)
        
        # 价值地图的衰减因子和更新率
        self.value_decay = 0.95  # 控制价值随距离的衰减速度
        self.value_update_rate = 0.1  # 控制价值地图的更新速率

        # Initial full and local pose
        self.full_pose = torch.zeros(self.num_scenes, 3).float().to(self.device)
        self.local_pose = torch.zeros(self.num_scenes, 3).float().to(self.device)

        # Origin of local map
        self.origins = np.zeros((self.num_scenes, 3))

        # Local Map Boundaries
        self.local_map_boundary = np.zeros((self.num_scenes, 4)).astype(int)

        # Planner pose inputs has 7 dimensions
        # 1-3 store continuous global agent location
        # 4-7 store local map boundaries
        self.planner_pose_inputs = np.zeros((self.num_scenes, 7))
    
    def mapping(self, obs, infos):
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        if obs.ndim == 3:
            obs = obs.unsqueeze(0)
        poses = torch.from_numpy(np.asarray(
            [infos['sensor_pose'] for env_idx
             in range(self.num_scenes)])
        ).float().to(self.device)
        agent_heights = torch.from_numpy(np.asarray(
            [infos['agent_height'] for env_idx in range(self.num_scenes)])
        ).float().to(self.device)

        _, self.local_map, _, self.local_pose = \
            self.mapping_module(obs, poses, self.local_map, self.local_pose, agent_heights)
        
        local_pose = self.local_pose.cpu().numpy()
        self.planner_pose_inputs[:, :3] = local_pose + self.origins
        self.local_map[:, 2, :, :].fill_(0.)  # Resetting current location channel
        for e in range(self.num_scenes):
            # r, c = locs[e, 1], locs[e, 0]
            self.local_row = int(local_pose[e, 1] * 100.0 / self.args.map_resolution)
            self.local_col = int(local_pose[e, 0] * 100.0 / self.args.map_resolution)
            self.local_map[e, 2:4, self.local_row - 2:self.local_row + 3, self.local_col - 2:self.local_col + 3] = 1.
    
    def move_local_map(self, env_idx=0):
        self.full_map[env_idx, :, self.local_map_boundary[env_idx, 0]:self.local_map_boundary[env_idx, 1], self.local_map_boundary[env_idx, 2]:self.local_map_boundary[env_idx, 3]] = \
            self.local_map[env_idx]
        # 同时更新价值地图
        self.full_value_map[env_idx, :, self.local_map_boundary[env_idx, 0]:self.local_map_boundary[env_idx, 1], self.local_map_boundary[env_idx, 2]:self.local_map_boundary[env_idx, 3]] = \
            self.local_value_map[env_idx]
        self.full_pose[env_idx] = self.local_pose[env_idx] + \
            torch.from_numpy(self.origins[env_idx]).to(self.device).float()

        locs = self.full_pose[env_idx].cpu().numpy()
        r, c = locs[1], locs[0]
        loc_r, loc_c = [int(r * 100.0 / self.args.map_resolution),
                        int(c * 100.0 / self.args.map_resolution)]

        self.local_map_boundary[env_idx] = self.get_local_map_boundaries((loc_r, loc_c))

        self.planner_pose_inputs[env_idx, 3:] = self.local_map_boundary[env_idx]
        self.origins[env_idx] = [self.local_map_boundary[env_idx][2] * self.args.map_resolution / 100.0,
                        self.local_map_boundary[env_idx][0] * self.args.map_resolution / 100.0, 0.]

        self.local_map[env_idx] = self.full_map[env_idx, :,
                                self.local_map_boundary[env_idx, 0]:self.local_map_boundary[env_idx, 1],
                                self.local_map_boundary[env_idx, 2]:self.local_map_boundary[env_idx, 3]]
        # 同时更新局部价值地图
        self.local_value_map[env_idx] = self.full_value_map[env_idx, :,
                                self.local_map_boundary[env_idx, 0]:self.local_map_boundary[env_idx, 1],
                                self.local_map_boundary[env_idx, 2]:self.local_map_boundary[env_idx, 3]]
        self.local_pose[env_idx] = self.full_pose[env_idx] - \
            torch.from_numpy(self.origins[env_idx]).to(self.device).float()
            
    def update_value_map(self, object_nodes=None, goal_position=None, env_idx=0):
        """
        更新价值地图，基于检测到的物体、目标位置和探索前沿
        
        Args:
            object_nodes: 场景图中的物体节点列表
            goal_position: 目标位置
            env_idx: 环境索引
        """
        # 使用当前的衰减率重置价值地图
        self.local_value_map[env_idx] = self.local_value_map[env_idx] * (1 - self.value_update_rate)
        
        # 根据物体位置添加价值
        if object_nodes is not None:
            for node in object_nodes:
                if hasattr(node, 'center') and node.center is not None:
                    # 将物体中心转换为局部地图坐标
                    x = int(node.center[0] - self.local_map_boundary[env_idx, 2])
                    y = int(node.center[1] - self.local_map_boundary[env_idx, 0])
                    
                    # 检查物体是否在局部地图范围内
                    if 0 <= x < self.local_width and 0 <= y < self.local_height:
                        # 物体重要性可以由其探索级别、得分或其他属性决定
                        importance = getattr(node, 'score', 0.5)
                        self._add_value_peak(x, y, importance, radius=5, env_idx=env_idx)
        
        # 为目标位置添加价值
        if goal_position is not None:
            x = int(goal_position[0] - self.local_map_boundary[env_idx, 2])
            y = int(goal_position[1] - self.local_map_boundary[env_idx, 0])
            
            if 0 <= x < self.local_width and 0 <= y < self.local_height:
                self._add_value_peak(x, y, 1.0, radius=7, env_idx=env_idx)
        
        # 为未探索区域添加价值（利用探索地图）
        exp_map = self.local_map[env_idx, 1, :, :].cpu().numpy()
        unexplored_mask = (exp_map < 0.1).astype(np.float32)
        
        # 在已探索和未探索的边界处添加价值
        # 这鼓励探索前沿
        kernel = np.ones((3, 3), np.float32)
        explored_mask = (exp_map > 0.5).astype(np.float32)
        dilated = cv2.dilate(explored_mask, kernel, iterations=2)
        frontier = dilated * unexplored_mask
        
        # 将前沿添加到价值地图
        frontier_tensor = torch.from_numpy(frontier).float().to(self.device)
        self.local_value_map[env_idx, 0] += frontier_tensor * 0.3
        
        # 将价值地图标准化到 [0, 1]
        self.local_value_map[env_idx] = torch.clamp(self.local_value_map[env_idx], 0.0, 1.0)
    
    def _add_value_peak(self, x, y, value, radius=5, env_idx=0):
        """
        在位置 (x, y) 处添加高斯峰值
        
        Args:
            x, y: 中心坐标
            value: 峰值
            radius: 影响半径
            env_idx: 环境索引
        """
        # 创建坐标网格
        y_coords, x_coords = torch.meshgrid(
            torch.arange(self.local_height, device=self.device),
            torch.arange(self.local_width, device=self.device)
        )
        
        # 计算与中心的平方距离
        sq_dist = (x_coords - x)**2 + (y_coords - y)**2
        
        # 高斯衰减
        falloff = torch.exp(-sq_dist / (2 * radius**2))
        
        # 将峰值添加到价值地图
        self.local_value_map[env_idx, 0] += falloff * value * self.value_update_rate
        
    def get_value_map(self, env_idx=0):
        """
        返回用于可视化或规划的局部价值地图
        
        Args:
            env_idx: 环境索引
            
        Returns:
            作为numpy数组的局部价值地图
        """
        return self.local_value_map[env_idx, 0].cpu().numpy()
        
    def init_map_and_pose(self):
        self.full_map.fill_(0.)
        self.full_pose.fill_(0.)
        self.full_pose[:, :2] = self.args.map_size_cm / 100.0 / 2.0
        
        # 初始化价值地图
        self.full_value_map.fill_(0.)

        locs = self.full_pose.cpu().numpy()
        self.planner_pose_inputs[:, :3] = locs
        for e in range(self.num_scenes):
            r, c = locs[e, 1], locs[e, 0]
            loc_r, loc_c = [int(r * 100.0 / self.args.map_resolution),
                            int(c * 100.0 / self.args.map_resolution)]

            self.full_map[e, 2:4, loc_r - 1:loc_r + 2, loc_c - 1:loc_c + 2] = 1.0

            self.local_map_boundary[e] = self.get_local_map_boundaries((loc_r, loc_c))

            self.planner_pose_inputs[e, 3:] = self.local_map_boundary[e]
            self.origins[e] = [self.local_map_boundary[e][2] * self.args.map_resolution / 100.0,
                          self.local_map_boundary[e][0] * self.args.map_resolution / 100.0, 0.]

        for e in range(self.num_scenes):
            self.local_map[e] = self.full_map[e, :,
                                    self.local_map_boundary[e, 0]:self.local_map_boundary[e, 1],
                                    self.local_map_boundary[e, 2]:self.local_map_boundary[e, 3]]
            # 同时更新局部价值地图
            self.local_value_map[e] = self.full_value_map[e, :,
                                    self.local_map_boundary[e, 0]:self.local_map_boundary[e, 1],
                                    self.local_map_boundary[e, 2]:self.local_map_boundary[e, 3]]
            self.local_pose[e] = self.full_pose[e] - \
                torch.from_numpy(self.origins[e]).to(self.device).float()

    def init_map_and_pose_for_env(self, env_idx=0):
        self.full_map[env_idx].fill_(0.)
        self.full_pose[env_idx].fill_(0.)
        self.full_pose[env_idx, :2] = self.args.map_size_cm / 100.0 / 2.0
        
        # 初始化价值地图
        self.full_value_map[env_idx].fill_(0.)

        locs = self.full_pose[env_idx].cpu().numpy()
        self.planner_pose_inputs[env_idx, :3] = locs
        r, c = locs[1], locs[0]
        loc_r, loc_c = [int(r * 100.0 / self.args.map_resolution),
                        int(c * 100.0 / self.args.map_resolution)]

        self.full_map[env_idx, 2:4, loc_r - 1:loc_r + 2, loc_c - 1:loc_c + 2] = 1.0

        self.local_map_boundary[env_idx] = self.get_local_map_boundaries((loc_r, loc_c))

        self.planner_pose_inputs[env_idx, 3:] = self.local_map_boundary[env_idx]
        self.origins[env_idx] = [self.local_map_boundary[env_idx][2] * self.args.map_resolution / 100.0,
                      self.local_map_boundary[env_idx][0] * self.args.map_resolution / 100.0, 0.]

        self.local_map[env_idx] = self.full_map[env_idx, :, self.local_map_boundary[env_idx, 0]:self.local_map_boundary[env_idx, 1], self.local_map_boundary[env_idx, 2]:self.local_map_boundary[env_idx, 3]]
        # 同时更新局部价值地图
        self.local_value_map[env_idx] = self.full_value_map[env_idx, :, self.local_map_boundary[env_idx, 0]:self.local_map_boundary[env_idx, 1], self.local_map_boundary[env_idx, 2]:self.local_map_boundary[env_idx, 3]]
        self.local_pose[env_idx] = self.full_pose[env_idx] - \
            torch.from_numpy(self.origins[env_idx]).to(self.device).float()

    def get_local_map_boundaries(self, agent_loc):
        loc_r, loc_c = agent_loc

        if self.args.global_downscaling > 1:
            gx1, gy1 = loc_r - self.local_width // 2, loc_c - self.local_height // 2
            gx2, gy2 = gx1 + self.local_width, gy1 + self.local_height
            if gx1 < 0:
                gx1, gx2 = 0, self.local_width
            if gx2 > self.global_width:
                gx1, gx2 = self.global_width - self.local_width, self.global_width

            if gy1 < 0:
                gy1, gy2 = 0, self.local_height
            if gy2 > self.global_height:
                gy1, gy2 = self.global_height - self.local_height, self.global_height
        else:
            gx1, gx2, gy1, gy2 = 0, self.global_width, 0, self.global_height

        return [gx1, gx2, gy1, gy2]
        
    def update_intrinsic_rew(self, env_idx=0):
        """
        更新内在奖励（intrinsic reward）
        同时将局部地图同步到全局地图
        
        Args:
            env_idx: 环境索引
        """
        self.full_map[env_idx, :, self.local_map_boundary[env_idx, 0]:self.local_map_boundary[env_idx, 1], self.local_map_boundary[env_idx, 2]:self.local_map_boundary[env_idx, 3]] = \
            self.local_map[env_idx]
        # 同时更新价值地图
        self.full_value_map[env_idx, :, self.local_map_boundary[env_idx, 0]:self.local_map_boundary[env_idx, 1], self.local_map_boundary[env_idx, 2]:self.local_map_boundary[env_idx, 3]] = \
            self.local_value_map[env_idx]