import warnings
warnings.filterwarnings('ignore')
import math
import os
import re
import cv2
from PIL import Image
import skimage.morphology
from skimage.draw import line_aa, line
import numpy as np
import torch
from torchvision import transforms

from src.utils.fmm.fmm_planner_policy import FMMPlanner
import src.utils.fmm.pose_utils as pu
from src.utils.visualization.semantic_prediction import SemanticPredMaskRCNN
from src.utils.visualization.visualization import (
    init_vis_image,
    draw_line,
    get_contour_points,
    line_list,
    add_text_list
)
from src.utils.visualization.save import save_video
from src.utils.llm import LLM

from lightglue import LightGlue, SuperPoint, DISK
from lightglue.utils import load_image, rbd, match_pair , numpy_image_to_torch



class UniGoal_Agent():
    def __init__(self, args, envs):
        self.args = args
        self.envs = envs
        self.device = args.device

        self.res = transforms.Compose(
            [transforms.ToPILImage(),
             transforms.Resize((args.frame_height, args.frame_width),
                               interpolation=Image.NEAREST)])


        self.sem_pred = SemanticPredMaskRCNN(args)
        self.llm = LLM(self.args.base_url, self.args.api_key, self.args.llm_model)

        self.selem = skimage.morphology.disk(3)

        self.rgbd = None
        self.obs_shape = None
        self.collision_map = None
        self.visited = None
        self.visited_vis = None
        self.col_width = None
        self.curr_loc = None
        self.last_loc = None
        self.last_action = None
        self.count_forward_actions = None
        self.instance_imagegoal = None
        self.text_goal = None

        self.extractor = DISK(max_num_keypoints=2048).eval().to(self.device)
        self.matcher = LightGlue(features='disk').eval().to(self.device)

        self.global_width = args.global_width
        self.global_height = args.global_height
        self.local_width = args.local_width
        self.local_height = args.local_height
        
        self.global_goal = None
        # define a temporal goal with a living time
        self.temp_goal = None
        self.last_temp_goal = None # avoid choose one goal twice
        self.forbidden_temp_goal = []
        self.flag = 0
        self.goal_instance_whwh = None
        # define untraversible area of the goal: 0 means area can be goals, 1 means cannot be
        self.goal_map_mask = np.ones((self.global_width, self.global_height))
        self.pred_box = []
        self.prompt_text2object = '"chair: 0, sofa: 1, plant: 2, bed: 3, toilet: 4, tv_monitor: 5" The above are the labels corresponding to each category. Which object is described in the following text? Only response the number of the label and not include other text.\nText: {text}'
        torch.set_grad_enabled(False)

        if args.visualize:
            self.vis_image_background = None
            self.rgb_vis = None
            self.vis_image_list = []

    def reset(self):
        args = self.args

        obs, info = self.envs.reset()

        if self.args.goal_type == 'ins-image':
            self.instance_imagegoal = self.envs.instance_imagegoal
        elif self.args.goal_type == 'text':
            self.text_goal = self.envs.text_goal
        idx = self.get_goal_cat_id()
        if idx is not None:
            self.envs.set_goal_cat_id(idx)

        rgbd = np.concatenate((obs['rgb'].astype(np.uint8), obs['depth']), axis=2).transpose(2, 0, 1)
        self.raw_obs = rgbd[:3, :, :].transpose(1, 2, 0)
        self.raw_depth = rgbd[3:4, :, :]

        rgbd, seg_predictions = self.preprocess_obs(rgbd)
        self.rgbd = rgbd

        self.obs_shape = rgbd.shape

        # Episode initializations
        map_shape = (args.map_size_cm // args.map_resolution,
                     args.map_size_cm // args.map_resolution)
        self.collision_map = np.zeros(map_shape)
        self.visited = np.zeros(map_shape)
        self.visited_vis = np.zeros(map_shape)
        self.col_width = 1
        self.count_forward_actions = 0
        self.curr_loc = [args.map_size_cm / 100.0 / 2.0,
                         args.map_size_cm / 100.0 / 2.0, 0.]
        self.last_action = None
        self.global_goal = None
        self.temp_goal = None
        self.last_temp_goal = None
        self.forbidden_temp_goal = []
        self.goal_map_mask = np.ones(map_shape)
        self.goal_instance_whwh = None
        self.pred_box = []
        self.been_stuck = False
        self.stuck_goal = None
        self.frontier_vis = None

        if args.visualize:
            self.vis_image_background = init_vis_image(self.envs.goal_name, self.args)

        return obs, rgbd, info

    def local_feature_match_lightglue(self, re_key2=False):
        with torch.set_grad_enabled(False):
            ob = numpy_image_to_torch(self.raw_obs[:, :, :3]).to(self.device)
            gi = numpy_image_to_torch(self.instance_imagegoal).to(self.device)
            try:
                feats0, feats1, matches01  = match_pair(self.extractor, self.matcher, ob, gi
                    )
                # indices with shape (K, 2)
                matches = matches01['matches']
                # in case that the matches collapse make a check
                b = torch.nonzero(matches[..., 0] < 2048, as_tuple=False)
                c = torch.index_select(matches[..., 0], dim=0, index=b.squeeze())
                points0 = feats0['keypoints'][c]
                if re_key2:
                    return (points0.numpy(), feats1['keypoints'][c].numpy())
                else:
                    return points0.numpy()  
            except:
                if re_key2:
                    # print(f'{self.env.rank}  {self.env.timestep}  h')
                    return (np.zeros((1, 2)), np.zeros((1, 2)))
                else:
                    # print(f'{self.env.rank}  {self.env.timestep}  h')
                    return np.zeros((1, 2))
                
    def compute_ins_dis_v1(self, depth, whwh, k=3):
        '''
        analyze the maxium depth points's pos
        make sure the object is within the range of 10m
        '''
        hist, bins = np.histogram(depth[whwh[1]:whwh[3], whwh[0]:whwh[2]].flatten(), \
            bins=200,range=(0,2000))
        peak_indices = np.argsort(hist)[-k:]  # Get the indices of the top k peaks
        peak_values = hist[peak_indices] + hist[np.clip(peak_indices-1, 0, len(hist)-1)]  + \
            hist[np.clip(peak_indices+1, 0, len(hist)-1)]
        max_area_index = np.argmax(peak_values)  # Find the index of the peak with the largest area
        max_index = peak_indices[max_area_index]
        # max_index = np.argmax(hist)
        return bins[max_index]

    def compute_ins_goal_map(self, whwh, start, start_o):
        goal_mask = np.zeros_like(self.rgbd[3, :, :])
        goal_mask[whwh[1]:whwh[3], whwh[0]:whwh[2]] = 1
        semantic_mask = (self.rgbd[4+self.envs.gt_goal_idx, :, :] > 0) & (goal_mask > 0)

        depth_h, depth_w = np.where(semantic_mask > 0)
        goal_dis = self.rgbd[3, :, :][depth_h, depth_w] / self.args.map_resolution

        goal_angle = -self.args.hfov / 2 * (depth_w - self.rgbd.shape[2]/2) \
        / (self.rgbd.shape[2]/2)
        goal = [start[0]+goal_dis*np.sin(np.deg2rad(start_o+goal_angle)), \
            start[1]+goal_dis*np.cos(np.deg2rad(start_o+goal_angle))]
        goal_map = np.zeros((self.local_width, self.local_height))
        goal[0] = np.clip(goal[0], 0, 240-1).astype(int)
        goal[1] = np.clip(goal[1], 0, 240-1).astype(int)
        goal_map[goal[0], goal[1]] = 1
        return goal_map

    def instance_discriminator(self, planner_inputs, id_lo_whwh_speci):
        # Get pose prediction and global policy planning window
        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = \
            planner_inputs['pose_pred']
        map_pred = np.rint(planner_inputs['map_pred'])
        gx1, gx2, gy1, gy2 = int(gx1), int(gx2), int(gy1), int(gy2)
        planning_window = [gx1, gx2, gy1, gy2]

        r, c = start_y, start_x
        start = [int(r * 100.0 / self.args.map_resolution - gx1),
                 int(c * 100.0 / self.args.map_resolution - gy1)]
        start = pu.threshold_poses(start, map_pred.shape)

        goal_mask = self.rgbd[4+self.envs.gt_goal_idx, :, :]

        if self.instance_imagegoal is None and self.text_goal is None:
            # not initialized
            return planner_inputs
        elif self.global_goal is not None:
            planner_inputs['found_goal'] = 1
            goal_map = pu.threshold_pose_map(self.global_goal, gx1, gx2, gy1, gy2)
            planner_inputs['goal'] = goal_map
            return planner_inputs
        elif self.been_stuck:
            
            planner_inputs['found_goal'] = 0
            if self.stuck_goal is None:

                navigable_indices = np.argwhere(self.visited[gx1:gx2, gy1:gy2] > 0)
                goal = np.array([0, 0])
                for _ in range(100):
                    random_index = np.random.choice(len(navigable_indices))
                    goal = navigable_indices[random_index]
                    if pu.get_l2_distance(goal[0], start[0], goal[1], start[1]) > 16:
                        break

                goal = pu.threshold_poses(goal, map_pred.shape)                
                self.stuck_goal = [int(goal[0])+gx1, int(goal[1])+gy1]
            else:
                goal = np.array([self.stuck_goal[0]-gx1, self.stuck_goal[1]-gy1])
                goal = pu.threshold_poses(goal, map_pred.shape)
            planner_inputs['goal'] = np.zeros((self.local_width, self.local_height))
            planner_inputs['goal'][int(goal[0]), int(goal[1])] = 1
        elif planner_inputs['found_goal'] == 1:
            id_lo_whwh_speci = sorted(id_lo_whwh_speci, 
                key=lambda s: (s[2][2]-s[2][0])**2+(s[2][3]-s[2][1])**2, reverse=True)
            whwh = (id_lo_whwh_speci[0][2] / 4).astype(int)
            w, h = whwh[2]-whwh[0], whwh[3]-whwh[1]
            goal_mask = np.zeros_like(goal_mask)
            goal_mask[whwh[1]:whwh[3], whwh[0]:whwh[2]] = 1.

            if self.args.goal_type == 'ins-image':
                index = self.local_feature_match_lightglue()
                match_points = index.shape[0]
            planner_inputs['found_goal'] = 0

            if self.temp_goal is not None:
                goal_map = pu.threshold_pose_map(self.temp_goal, gx1, gx2, gy1, gy2)
                goal_dis = self.compute_temp_goal_distance(map_pred, goal_map, start, planning_window)
            else:
                goal_map = self.compute_ins_goal_map(whwh, start, start_o)
                if not np.any(goal_map>0) :
                    tgoal_dis = self.compute_ins_dis_v1(self.rgbd[3, :, :], whwh) / self.args.map_resolution
                    rgb_center = np.array([whwh[3]+whwh[1], whwh[2]+whwh[0]])//2
                    goal_angle = -self.args.hfov / 2 * (rgb_center[1] - self.rgbd.shape[2]/2) \
                    / (self.rgbd.shape[2]/2)
                    goal = [start[0]+tgoal_dis*np.sin(np.deg2rad(start_o+goal_angle)), \
                        start[1]+tgoal_dis*np.cos(np.deg2rad(start_o+goal_angle))]
                    goal = pu.threshold_poses(goal, map_pred.shape)
                    rr,cc = skimage.draw.ellipse(goal[0], goal[1], 10, 10, shape=goal_map.shape)
                    goal_map[rr, cc] = 1


                goal_dis = self.compute_temp_goal_distance(map_pred, goal_map, start, planning_window)

            if goal_dis is None:
                self.temp_goal = None
                planner_inputs['goal'] = planner_inputs['exp_goal']
                selem = skimage.morphology.disk(3)
                goal_map = skimage.morphology.dilation(goal_map, selem)
                self.goal_map_mask[gx1:gx2, gy1:gy2][goal_map > 0] = 0
                print(f"Rank: {self.envs.rank}, timestep: {self.envs.timestep},  temp goal unavigable !")
            else:
                if self.args.goal_type == 'ins-image' and match_points > 100:
                    planner_inputs['found_goal'] = 1
                    global_goal = np.zeros((self.global_width, self.global_height))
                    global_goal[gx1:gx2, gy1:gy2] = goal_map
                    self.global_goal = global_goal
                    planner_inputs['goal'] = goal_map
                    self.temp_goal = None
                else:
                    if (self.args.goal_type == 'ins-image' and goal_dis < 50) or (self.args.goal_type == 'text' and goal_dis < 15):
                        if (self.args.goal_type == 'ins-image' and match_points > 90) or self.args.goal_type == 'text':
                            planner_inputs['found_goal'] = 1
                            global_goal = np.zeros((self.global_width, self.global_height))
                            global_goal[gx1:gx2, gy1:gy2] = goal_map
                            self.global_goal = global_goal
                            planner_inputs['goal'] = goal_map
                            self.temp_goal = None
                        else:
                            planner_inputs['goal'] = planner_inputs['exp_goal']
                            self.temp_goal = None
                            selem = skimage.morphology.disk(1)
                            goal_map = skimage.morphology.dilation(goal_map, selem)
                            self.goal_map_mask[gx1:gx2, gy1:gy2][goal_map > 0] = 0
                    else:
                        new_goal_map = goal_map * self.goal_map_mask[gx1:gx2, gy1:gy2]
                        if np.any(new_goal_map > 0):
                            planner_inputs['goal'] = new_goal_map
                            temp_goal = np.zeros((self.global_width, self.global_height))
                            temp_goal[gx1:gx2, gy1:gy2] = new_goal_map
                            self.temp_goal = temp_goal
                        else:
                            planner_inputs['goal'] = planner_inputs['exp_goal']
                            self.temp_goal = None
            return planner_inputs

        else:
            planner_inputs['goal'] = planner_inputs['exp_goal']
            if self.temp_goal is not None:  
                goal_map = pu.threshold_pose_map(self.temp_goal, gx1, gx2, gy1, gy2)
                goal_dis = self.compute_temp_goal_distance(map_pred, goal_map, start, planning_window)
                planner_inputs['found_goal'] = 0
                new_goal_map = goal_map * self.goal_map_mask[gx1:gx2, gy1:gy2]
                if np.any(new_goal_map > 0):
                    if goal_dis is not None:
                        planner_inputs['goal'] = new_goal_map
                        if goal_dis < 100:
                            if self.args.goal_type == 'ins-image':
                                index = self.local_feature_match_lightglue()
                                match_points = index.shape[0]
                            if (self.args.goal_type == 'ins-image' and match_points < 80) or self.args.goal_type == 'text':
                                planner_inputs['goal'] = planner_inputs['exp_goal']
                                selem = skimage.morphology.disk(3)
                                new_goal_map = skimage.morphology.dilation(new_goal_map, selem)
                                self.goal_map_mask[gx1:gx2, gy1:gy2][new_goal_map > 0] = 0
                                self.temp_goal = None
                    else:
                        selem = skimage.morphology.disk(3)
                        new_goal_map = skimage.morphology.dilation(new_goal_map, selem)
                        self.goal_map_mask[gx1:gx2, gy1:gy2][new_goal_map > 0] = 0
                        self.temp_goal = None
                        print(f"Rank: {self.envs.rank}, timestep: {self.envs.timestep},  temp goal unavigable !")
                else:
                    self.temp_goal = None
                    
                    
            return planner_inputs


    def step(self, agent_input):
        """Function responsible for planning, taking the action and
        preprocessing observations

        Args:
            planner_inputs (dict):
                dict with following keys:
                    'map_pred'  (ndarray): (M, M) map prediction
                    'goal'      (ndarray): (M, M) mat denoting goal locations
                    'pose_pred' (ndarray): (7,) array denoting pose (x,y,o)
                                 and planning window (gx1, gx2, gy1, gy2)
                     'found_goal' (bool): whether the goal object is found

        Returns:
            obs (ndarray): preprocessed observations ((4+C) x H x W)
            reward (float): amount of reward returned after previous action
            done (bool): whether the episode has ended
            info (dict): contains timestep, pose, goal category and
                         evaluation metric info
        """

        # plan
        if agent_input["wait"]:
            self.last_action = None
            self.envs.info["sensor_pose"] = [0., 0., 0.]
            return None, np.zeros(self.rgbd.shape), False, self.envs.info

        id_lo_whwh = self.pred_box


        id_lo_whwh_speci = [id_lo_whwh[i] for i in range(len(id_lo_whwh)) \
                    if id_lo_whwh[i][0] == self.envs.gt_goal_idx]


        agent_input["found_goal"] = (id_lo_whwh_speci != [])

        self.instance_discriminator(agent_input, id_lo_whwh_speci)

        action = self.get_action(agent_input)

        if self.args.visualize:
            self.visualize(agent_input)

        if action >= 0:
            action = {'action': action}
            obs, done, info = self.envs.step(action)
            rgbd = np.concatenate((obs['rgb'].astype(np.uint8), obs['depth']), axis=2).transpose(2, 0, 1)
            self.raw_obs = rgbd[:3, :, :].transpose(1, 2, 0)
            self.raw_depth = rgbd[3:4, :, :]

            rgbd, seg_predictions = self.preprocess_obs(rgbd) 
            self.last_action = action['action']
            self.rgbd = rgbd

            if done:
                obs, rgbd, info = self.reset()

            return obs, rgbd, done, self.envs.info

        else:
            self.last_action = None
            self.envs.info["sensor_pose"] = [0., 0., 0.]
            return None, np.zeros(self.obs_shape), False, self.envs.info

    def get_action(self, planner_inputs):
        """Function responsible for planning

        Args:
            planner_inputs (dict):
                dict with following keys:
                    'map_pred'  (ndarray): (M, M) map prediction
                    'goal'      (ndarray): (M, M) goal locations
                    'pose_pred' (ndarray): (7,) array  denoting pose (x,y,o)
                                 and planning window (gx1, gx2, gy1, gy2)
                    'found_goal' (bool): whether the goal object is found

        Returns:
            action (int): action id
        """
        args = self.args

        self.last_loc = self.curr_loc

        # 获取地图预测和构建障碍物网格
        grid = 1 - np.rint(planner_inputs['map_pred'])
        # 概率转换为对数比率（log-odds）
        grid = grid.astype(np.float32)
        
        # 应用价值地图影响路径规划（如果提供了价值地图）
        if 'value_map' in planner_inputs:
            # 将价值地图缩放到 [0, 0.5] 范围，
            # 这确保它可以影响但不会完全覆盖可通行性
            value_influence = planner_inputs['value_map'] * 0.5
            # 将价值地图的影响应用到网格上（在高价值区域减少成本）
            grid = grid - value_influence
            # 确保网格值保持在有效范围 [0, 1] 内
            grid = np.clip(grid, 0.0, 1.0)

        # 获取姿态预测和全局策略规划窗口
        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = \
            planner_inputs['pose_pred']
        gx1, gx2, gy1, gy2 = int(gx1), int(gx2), int(gy1), int(gy2)
        planning_window = [gx1, gx2, gy1, gy2]

        # 获取目标
        goal = planner_inputs['goal']

        # 获取当前位置
        self.curr_loc = [start_x, start_y, start_o]
        r, c = start_y, start_x
        start = [int(r * 100.0 / args.map_resolution - gx1),
                 int(c * 100.0 / args.map_resolution - gy1)]
        start = pu.threshold_poses(start, grid.shape)
        
        # 获取上一个位置
        last_start_x, last_start_y = self.last_loc[0], self.last_loc[1]
        r, c = last_start_y, last_start_x
        last_start = [int(r * 100.0 / args.map_resolution - gx1),
                        int(c * 100.0 / args.map_resolution - gy1)]
        last_start = pu.threshold_poses(last_start, grid.shape)
        
        rr, cc, _ = line_aa(last_start[0], last_start[1], start[0], start[1])
        self.visited[gx1:gx2, gy1:gy2][rr, cc] += 1

        if args.visualize:            
            self.visited_vis[gx1:gx2, gy1:gy2] = \
                draw_line(last_start, start,
                             self.visited_vis[gx1:gx2, gy1:gy2])

        # 处理卡住目标的问题
        x1, y1, t1 = self.last_loc
        x2, y2, _ = self.curr_loc
        if abs(x1 - x2) >= 0.05 or abs(y1 - y2) >= 0.05:
            self.been_stuck = False
            self.stuck_goal = None

        # 碰撞检查
        if self.last_action == 1:
            x1, y1, t1 = self.last_loc
            x2, y2, _ = self.curr_loc
            buf = 4
            length = 2

            if abs(x1 - x2) < 0.05 and abs(y1 - y2) < 0.05:
                self.col_width += 2
                if self.col_width == 7:
                    length = 4
                    buf = 3
                    self.been_stuck = True
                self.col_width = min(self.col_width, 5)
            else:
                self.col_width = 1                

            dist = pu.get_l2_distance(x1, x2, y1, y2)
            if dist < args.collision_threshold:  # 碰撞
                width = self.col_width
                for i in range(length):
                    for j in range(width):
                        wx = x1 + 0.05 * \
                            ((i + buf) * np.cos(np.deg2rad(t1))
                             + (j - width // 2) * np.sin(np.deg2rad(t1)))
                        wy = y1 + 0.05 * \
                            ((i + buf) * np.sin(np.deg2rad(t1))
                             - (j - width // 2) * np.cos(np.deg2rad(t1)))
                        r, c = wy, wx
                        r, c = int(r * 100 / args.map_resolution), \
                            int(c * 100 / args.map_resolution)
                        [r, c] = pu.threshold_poses([r, c],
                                                    self.collision_map.shape)
                        self.collision_map[r, c] = 1

        # 存储规划输入，供 get_local_goal 使用
        self.planner_inputs = planner_inputs
        
        # 获取局部目标
        local_goal, stop = self.get_local_goal(grid, start, np.copy(goal),
                                  planning_window)

        if stop and planner_inputs['found_goal'] == 1:
            action = 0
        else:
            (local_x, local_y) = local_goal
            angle_st_goal = math.degrees(math.atan2(local_x - start[0],
                                                    local_y - start[1]))
            angle_agent = (start_o) % 360.0
            if angle_agent > 180:
                angle_agent -= 360

            relative_angle = (angle_agent - angle_st_goal) % 360.0
            if relative_angle > 180:
                relative_angle -= 360

            if relative_angle > self.args.turn_angle:
                action = 3
            elif relative_angle < -self.args.turn_angle:
                action = 2
            else:
                action = 1

        return action

    def get_local_goal(self, grid, start, goal, planning_window):
        """
        获取到达全局目标的局部目标
        同时考虑可达性和位置价值
        """
        [gx1, gx2, gy1, gy2] = planning_window
        H, W = grid.shape
        start = pu.threshold_poses(start, grid.shape)
        
        goal_map = goal.copy()
        
        # 如果可用，使用价值地图影响目标选择
        # 当有多条路径可用时，应该优先考虑高价值位置
        value_weight = 0.3  # 价值地图影响的权重
        
        if hasattr(self, 'planner_inputs') and 'value_map' in self.planner_inputs:
            value_map = self.planner_inputs['value_map']
            
            # 如果有多个目标候选，基于价值进行优先排序
            if goal_map.sum() > 1:
                goal_y, goal_x = np.where(goal_map > 0)
                if len(goal_y) > 0:
                    # 获取每个目标候选的价值
                    goal_values = [value_map[y, x] for y, x in zip(goal_y, goal_x)]
                    # 选择价值最高的目标
                    max_value_idx = np.argmax(goal_values)
                    new_goal_map = np.zeros_like(goal_map)
                    new_goal_map[goal_y[max_value_idx], goal_x[max_value_idx]] = 1
                    goal_map = new_goal_map
        
        # 获取地图中有效的、已探索的区域
        traversible = grid.copy()
        
        # 创建FMM规划器并设置起始位置和可通行区域
        from src.utils.fmm.fmm_planner import FMMPlanner
        planner = FMMPlanner(traversible)
        
        # 找到目标位置
        goal_y, goal_x = np.where(goal_map > 0)
        if len(goal_y) == 0:
            # 如果没有目标，停止
            return start, True
            
        # 选择第一个目标点
        goal_pos = [goal_x[0], goal_y[0]]
        
        # 设置目标
        planner.set_goal(goal_pos)
        
        # 获取从起点到目标的最短路径
        stg_x, stg_y = start
        stg_x, stg_y, replan, stop = planner.get_short_term_goal(start)
        
        # 检查是否到达目标
        if stop:
            return start, True
        
        # 返回局部目标
        return (int(stg_x), int(stg_y)), False

    def add_boundary(self, mat, value=1):
        h, w = mat.shape
        new_mat = np.zeros((h + 2, w + 2)) + value
        new_mat[1:h + 1, 1:w + 1] = mat
        return new_mat

    def compute_temp_goal_distance(self, grid, goal_map, start, planning_window):
        [gx1, gx2, gy1, gy2] = planning_window
        x1, y1, = (
            0,
            0,
        )
        x2, y2 = grid.shape
        goal = goal_map * 1
        traversible = 1.0 - cv2.dilate(grid[x1:x2, y1:y2], self.selem)
        traversible[self.visited[gx1:gx2, gy1:gy2][x1:x2, y1:y2] > 0] = 1
        traversible[self.collision_map[gx1:gx2, gy1:gy2][x1:x2, y1:y2] == 1] = 0
        
        traversible[int(start[0] - x1) - 1:int(start[0] - x1) + 2,
                    int(start[1] - y1) - 1:int(start[1] - y1) + 2] = 1

        st_dis = pu.get_l2_dis_point_map(start, goal) * self.args.map_resolution  # cm

        traversible = self.add_boundary(traversible)
        planner = FMMPlanner(traversible)
        selem = skimage.morphology.disk(10)
        
        goal = cv2.dilate(goal, selem)
        
        goal = self.add_boundary(goal, value=0)
        
        # 找到目标位置坐标
        goal_y, goal_x = np.where(goal > 0)
        if len(goal_y) == 0:
            return None
            
        # 创建一个目标点
        goal_pos = [int(goal_x[0]), int(goal_y[0])]
        
        # 设置目标位置
        planner.set_goal(goal_pos)
        
        fmm_dist = planner.fmm_dist * self.args.map_resolution 
        dis = fmm_dist[start[0]+1, start[1]+1]

        if dis < fmm_dist.max() and dis/st_dis < 2:
            return dis
        else:
            return None

    def preprocess_obs(self, obs, use_seg=True):
        args = self.args
        obs = obs.transpose(1, 2, 0)
        rgb = obs[:, :, :3]
        depth = obs[:, :, 3:4]

        sem_seg_pred, seg_predictions = self.pred_sem(
            rgb.astype(np.uint8), use_seg=use_seg)

        if args.environment == 'habitat':
            depth = self.preprocess_depth(depth, args.min_depth, args.max_depth)

        ds = args.env_frame_width // args.frame_width  # Downscaling factor
        if ds != 1:
            rgb = np.asarray(self.res(rgb.astype(np.uint8)))
            depth = depth[ds // 2::ds, ds // 2::ds]
            sem_seg_pred = sem_seg_pred[ds // 2::ds, ds // 2::ds]

        depth = np.expand_dims(depth, axis=2)
        state = np.concatenate((rgb, depth, sem_seg_pred),
                               axis=2).transpose(2, 0, 1)

        return state, seg_predictions

    def preprocess_depth(self, depth, min_d, max_d):
        depth = depth[:, :, 0] * 1

        for i in range(depth.shape[0]):
            depth[i, :][depth[i, :] == 0.] = depth[i, :].max() + 0.01

        mask2 = depth > 0.99
        depth[mask2] = 0.

        mask1 = depth == 0
        depth[mask1] = 100.0
        depth = min_d * 100.0 + depth * max_d * 100.0
        return depth

    def pred_sem(self, rgb, depth=None, use_seg=True, pred_bbox=False):
        if pred_bbox:
            semantic_pred, self.rgb_vis, self.pred_box, seg_predictions = self.sem_pred.get_prediction(rgb)
            return self.pred_box, seg_predictions
        else:
            if use_seg:
                semantic_pred, self.rgb_vis, self.pred_box, seg_predictions = self.sem_pred.get_prediction(rgb)
                semantic_pred = semantic_pred.astype(np.float32)
                if depth is not None:
                    normalize_depth = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    self.rgb_vis = cv2.cvtColor(normalize_depth, cv2.COLOR_GRAY2BGR)
            else:
                semantic_pred = np.zeros((rgb.shape[0], rgb.shape[1], 16))
                self.rgb_vis = rgb[:, :, ::-1]
            return semantic_pred, seg_predictions
        
    def get_goal_cat_id(self):
        if self.args.goal_type == 'ins-image':
            instance_whwh, seg_predictions = self.pred_sem(self.instance_imagegoal.astype(np.uint8), None, pred_bbox=True)
            ins_whwh = [instance_whwh[i] for i in range(len(instance_whwh)) \
                if (instance_whwh[i][2][3]-instance_whwh[i][2][1])>1/6*self.instance_imagegoal.shape[0] or \
                    (instance_whwh[i][2][2]-instance_whwh[i][2][0])>1/6*self.instance_imagegoal.shape[1]]
            if ins_whwh != []:
                ins_whwh = sorted(ins_whwh,  \
                    key=lambda s: ((s[2][0]+s[2][2]-self.instance_imagegoal.shape[1])/2)**2 \
                        +((s[2][1]+s[2][3]-self.instance_imagegoal.shape[0])/2)**2 \
                    )
                if ((ins_whwh[0][2][0]+ins_whwh[0][2][2]-self.instance_imagegoal.shape[1])/2)**2 \
                        +((ins_whwh[0][2][1]+ins_whwh[0][2][3]-self.instance_imagegoal.shape[0])/2)**2 < \
                            ((self.instance_imagegoal.shape[1] / 6)**2 )*2:
                    return int(ins_whwh[0][0])
            return None
        elif self.args.goal_type == 'text':
            for i in range(10):
                if isinstance(self.text_goal, dict) and 'intrinsic_attributes' in self.text_goal:  
                    text_goal = self.text_goal['intrinsic_attributes']
                else:
                    text_goal = self.text_goal
                text_goal_id = self.llm(self.prompt_text2object.replace('{text}', text_goal))
                try:
                    text_goal_id = re.findall(r'\d+', text_goal_id)[0]
                    text_goal_id = int(text_goal_id)
                    if 0 <= text_goal_id < 6:
                        return text_goal_id
                except:
                    pass
            return 0

    def visualize(self, inputs):
        args = self.args

        color_palette = [
            # 原来的调色板注释掉
            # 1.0, 1.0, 1.0,
            # 0.6, 0.6, 0.6,
            # 0.95, 0.95, 0.95,
            # 0.96, 0.36, 0.26,
            # 0.12156862745098039, 0.47058823529411764, 0.7058823529411765,
            # 0.9400000000000001, 0.7818, 0.66,
            # 0.8882000000000001, 0.9400000000000001, 0.66,
            # 0.66, 0.9400000000000001, 0.8518000000000001,
            # 0.7117999999999999, 0.66, 0.9400000000000001,
            # 0.9218, 0.66, 0.9400000000000001,
            # 0.9400000000000001, 0.66, 0.748199999999999
            
            # 新的更鲜明的调色板
            1.0, 1.0, 1.0,      # 0: 白色 - 背景
            0.2, 0.2, 0.2,      # 1: 深灰色 - 障碍物
            0.9, 0.9, 0.9,      # 2: 浅灰色 - 已探索区域
            0.0, 0.8, 0.0,      # 3: 绿色 - 访问路径
            1.0, 0.0, 0.0,      # 4: 红色 - 目标区域
            0.0, 0.5, 1.0,      # 5: 蓝色 - 类别1
            1.0, 0.6, 0.0,      # 6: 橙色 - 类别2
            0.8, 0.0, 0.8,      # 7: 紫色 - 类别3
            1.0, 1.0, 0.0,      # 8: 黄色 - 类别4
            0.0, 0.8, 0.8,      # 9: 青色 - 类别5
            0.5, 1.0, 0.5,      # 10: 浅绿色 - 类别6
            0.7, 0.3, 0.0       # 11: 棕色 - 类别7
        ]

        map_pred = inputs['map_pred']
        exp_pred = inputs['exp_pred']
        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = inputs['pose_pred']

        goal = inputs['goal']
        sem_map = inputs['sem_map_pred']

        gx1, gx2, gy1, gy2 = int(gx1), int(gx2), int(gy1), int(gy2)

        # add a check with collision map
        map_pred[self.collision_map[gx1:gx2, gy1:gy2] == 1] = 1

        sem_map += 5

        no_cat_mask = sem_map == 11
        # no_cat_mask = np.logical_or(no_cat_mask, 1 - no_cat_mask)
        map_mask = np.rint(map_pred) == 1
        exp_mask = np.rint(exp_pred) == 1
        vis_mask = self.visited_vis[gx1:gx2, gy1:gy2] == 1
        # vis_mask = self.visited[gx1:gx2, gy1:gy2] == 1

        sem_map[no_cat_mask] = 0
        m1 = np.logical_and(no_cat_mask, exp_mask)
        sem_map[m1] = 2

        m2 = np.logical_and(no_cat_mask, map_mask)
        sem_map[m2] = 1

        sem_map[vis_mask] = 3

        # <goal>
        selem = skimage.morphology.disk(4)
        goal_mat = 1 - skimage.morphology.binary_dilation(
            goal, selem) != True

        goal_mask = goal_mat == 1
        sem_map[goal_mask] = 4
        # </goal>

        if self.args.environment == 'habitat':
            locs = np.array(self.envs._env.current_episode.goals[0].position[:2]) + np.array([18, 18])
            r, c = locs[1], locs[0]
            loc_r, loc_c = [int(r * 100.0 / args.map_resolution),
                            int(c * 100.0 / args.map_resolution)]

            if gx1 + 1 <= loc_c < gx2 - 1 and gy1 + 1 <= loc_r < gy2 - 1:
                sem_map[loc_r - gy1 - 1:loc_r - gy1 + 2, loc_c - gx1 - 1:loc_c - gx1 + 2] = [255, 0, 0]

        color_pal = [int(x * 255.) for x in color_palette]
        sem_map_vis = Image.new("P", (sem_map.shape[1],
                                      sem_map.shape[0]))
        sem_map_vis.putpalette(color_pal)
        sem_map_vis.putdata(sem_map.flatten().astype(np.uint8))
        sem_map_vis = sem_map_vis.convert("RGB")
        sem_map_vis = np.flipud(sem_map_vis)

        sem_map_vis = sem_map_vis[:, :, [2, 1, 0]]
        # sem_map_vis = insert_s_goal(self.s_goal, sem_map_vis, goal)
        sem_map_vis = cv2.resize(sem_map_vis, (480, 480),
                                 interpolation=cv2.INTER_NEAREST)
        if self.args.environment == 'habitat':
            rgb_visualization = cv2.resize(self.rgb_vis, (360, 480), interpolation=cv2.INTER_NEAREST)

        vis_image = self.vis_image_background.copy()
        
        # 可视化价值地图
        if 'value_map' in inputs:
            value_map = inputs['value_map']
            # 创建热力图可视化
            value_map_vis = cv2.normalize(value_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            value_map_color = cv2.applyColorMap(value_map_vis, cv2.COLORMAP_JET)
            value_map_vis = cv2.resize(value_map_color, (240, 240), interpolation=cv2.INTER_NEAREST)
            
            # 将价值地图绘制在可视化界面右侧（改变位置，避免遮挡左上角文字）
            vis_image[50:290, 1140:1380] = value_map_vis
            cv2.putText(vis_image, "Value Map", (1140, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # 添加价值地图的图例
            cv2.rectangle(vis_image, (1140, 300), (1380, 320), (0, 0, 0), -1)
            for i in range(240):
                color = cv2.applyColorMap(np.array([[int(i * 255 / 240)]], dtype=np.uint8), cv2.COLORMAP_JET)[0, 0]
                cv2.line(vis_image, (1140 + i, 300), (1140 + i, 320), (int(color[0]), int(color[1]), int(color[2])), 1)
            cv2.putText(vis_image, "Low", (1140, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(vis_image, "High", (1340, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        if self.args.goal_type == 'ins-image':
            instance_imagegoal = self.instance_imagegoal
            h, w = instance_imagegoal.shape[0], instance_imagegoal.shape[1]
            if h > w:
                instance_imagegoal = instance_imagegoal[h // 2 - w // 2:h // 2 + w // 2, :]
            elif w > h:
                instance_imagegoal = instance_imagegoal[:, w // 2 - h // 2:w // 2 + h // 2]
            instance_imagegoal = cv2.resize(instance_imagegoal, (215, 215), interpolation=cv2.INTER_NEAREST)
            instance_imagegoal = cv2.cvtColor(instance_imagegoal, cv2.COLOR_RGB2BGR)
            if self.args.environment == 'habitat':
                vis_image[50:265, 25:240] = instance_imagegoal
        elif self.args.goal_type == 'text':
            if isinstance(self.text_goal, dict) and 'intrinsic_attributes' in self.text_goal and 'extrinsic_attributes' in self.text_goal:
                text_goal = self.text_goal['intrinsic_attributes'] + ' ' + self.text_goal['extrinsic_attributes']
            else:
                text_goal = self.text_goal
            text_goal = line_list(text_goal)[:12]
            add_text_list(vis_image[50:265, 25:240], text_goal)
        vis_image[50:530, 650:1130] = sem_map_vis
        if self.args.environment == 'habitat':
            vis_image[50:530, 265:625] = rgb_visualization
        if self.args.environment == 'habitat':
            cv2.rectangle(vis_image, (25, 50), (240, 265), (0, 0, 255), 2)  # 红色，更粗的线
            cv2.rectangle(vis_image, (25, 315), (240, 530), (0, 0, 255), 2)  # 红色，更粗的线
        cv2.rectangle(vis_image, (650, 50), (1130, 530), (0, 0, 255), 2)  # 红色，更粗的线
        if self.args.environment == 'habitat':
            cv2.rectangle(vis_image, (265, 50), (625, 530), (0, 0, 255), 2)  # 红色，更粗的线

        pos = (
            (start_x * 100. / args.map_resolution - gy1)
            * 480 / map_pred.shape[0],
            (map_pred.shape[1] - start_y * 100. / args.map_resolution + gx1)
            * 480 / map_pred.shape[1],
            np.deg2rad(-start_o)
        )

        agent_arrow = get_contour_points(pos, origin=(885-200-10-25, 50))
        color = (0, 0, 255)
        cv2.drawContours(vis_image, [agent_arrow], 0, color, -1)

        self.vis_image_list.append(vis_image)
        tmp_dir = 'outputs/tmp'
        os.makedirs(tmp_dir, exist_ok=True)
        height, width, layers = vis_image.shape
        if self.args.is_debugging:
            image_name = 'debug.jpg'
        else:
            image_name = 'v.jpg'
        cv2.imwrite(os.path.join(tmp_dir, image_name), cv2.resize(vis_image, (width // 2, height // 2)))
    
    def save_visualization(self, video_path):
        save_video(self.vis_image_list, video_path, fps=15, input_color_space="BGR")
        self.vis_image_list = []
