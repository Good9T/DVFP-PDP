import torch
import numpy as np
import pandas as pd
import copy


def get_random_problems(batch_size, customer_size, problem_gen_params):
    scaler = problem_gen_params['scaler']
    depot_size = 1
    node_num = depot_size + customer_size * 2  # depot + pick + delivery

    # 生成原始节点坐标：depot, pick, delivery
    depot_xy = torch.rand(batch_size, depot_size, 2)
    pick_xy = torch.rand(batch_size, customer_size, 2)
    delivery_xy = torch.rand(batch_size, customer_size, 2)

    # ======================
    # Primal 原始视角：[depot, P, D]
    # ======================
    node_coords = torch.cat([depot_xy, pick_xy, delivery_xy], dim=1)

    coord_diff = node_coords.unsqueeze(2) - node_coords.unsqueeze(1)
    straight_dist = torch.norm(coord_diff, p=2, dim=-1)
    straight_dist[:, torch.arange(node_num), torch.arange(node_num)] = 0.0
    problems = straight_dist.float() / scaler

    # ======================
    # Dual 对偶视角：取送交换 [depot, D, P]
    # ======================
    node_coords_dual = torch.cat([depot_xy, delivery_xy, pick_xy], dim=1)

    coord_diff_dual = node_coords_dual.unsqueeze(2) - node_coords_dual.unsqueeze(1)
    straight_dist_dual = torch.norm(coord_diff_dual, p=2, dim=-1)
    straight_dist_dual[:, torch.arange(node_num), torch.arange(node_num)] = 0.0
    dual_problems = straight_dist_dual.float() / scaler

    return problems, dual_problems, node_coords


def get_dataset_problem(load_path, batch_size, aug_type='8'):
    filename = load_path
    data = pd.read_csv(filename, sep=',', header=None)
    data = data.to_numpy()
    depot_size = 1
    customer_size = int(data[0][0])
    scale = int(data[0][1])
    depot_xy = data[1:depot_size + 1] / scale
    pick_xy = data[depot_size + 1:depot_size + customer_size + 1] / scale
    delivery_xy = data[depot_size + customer_size + 1:depot_size + 2 * customer_size + 1] / scale
    full_node = data[1:depot_size + 2 * customer_size + 1] / scale

    depot_x_y = torch.FloatTensor(depot_xy[0][0:2]).unsqueeze(0)
    for i in range(len(2 * pick_xy)):
        pick_x_y = torch.FloatTensor(pick_xy[i][0:2]).unsqueeze(0) if i == 0 else torch.cat(
            [pick_x_y, torch.FloatTensor(pick_xy[i][0:2]).unsqueeze(0)], dim=0)
        delivery_x_y = torch.FloatTensor(delivery_xy[i][0:2]).unsqueeze(0) if i == 0 else torch.cat(
            [delivery_x_y, torch.FloatTensor(delivery_xy[i][0:2]).unsqueeze(0)], dim=0)
    depot_x_y = depot_x_y.unsqueeze(0).repeat(batch_size, 1, 1)
    pick_x_y = pick_x_y.unsqueeze(0).repeat(batch_size, 1, 1)
    delivery_x_y = delivery_x_y.unsqueeze(0).repeat(batch_size, 1, 1)
    depot_x_y, pick_x_y, delivery_x_y, aug_number = aug(
        aug_type=aug_type, depot_x_y=depot_x_y, pick_x_y=pick_x_y, delivery_x_y=delivery_x_y)
    data = {'depot_x_y': depot_x_y.numpy().tolist(), 'pick_x_y': pick_x_y.numpy().tolist(),
            'delivery_x_y': delivery_x_y.numpy().tolist(),
            'full_node': full_node, 'scale': scale, 'aug_number': aug_number}
    return depot_x_y, pick_x_y, delivery_x_y, customer_size, data, aug_number
