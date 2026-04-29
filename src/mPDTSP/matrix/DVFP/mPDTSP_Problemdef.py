import torch
import numpy as np
import pandas as pd

def get_random_problems(batch_size, customer_size, problem_gen_params):
    scaler = problem_gen_params['scaler']
    capacity = problem_gen_params['capacity']
    demand_min = problem_gen_params['demand_min']
    demand_max = problem_gen_params['demand_max']

    depot_size = 1
    node_num = depot_size + customer_size * 2

    # 坐标生成
    depot_xy = torch.rand(batch_size, depot_size, 2)
    pick_xy = torch.rand(batch_size, customer_size, 2)
    delivery_xy = torch.rand(batch_size, customer_size, 2)

    # 需求
    pick_demand = torch.randint(demand_min, demand_max, (batch_size, customer_size))
    delivery_demand = -pick_demand

    pick_demand = pick_demand / capacity
    delivery_demand = delivery_demand / capacity

    # 拼接完整需求 (Env 要用这个！)
    depot_demand = torch.zeros(batch_size, 1)
    node_demand = torch.cat([depot_demand, pick_demand, delivery_demand], dim=1)  # (B, N)

    # ===================== Primal / Dual 坐标 =====================
    node_coords = torch.cat([depot_xy, pick_xy, delivery_xy], dim=1)
    node_coords_dual = torch.cat([depot_xy, delivery_xy, pick_xy], dim=1)

    # ===================== 距离矩阵 =====================
    # Primal
    coord_diff = node_coords.unsqueeze(2) - node_coords.unsqueeze(1)
    dist = torch.norm(coord_diff, p=2, dim=-1)
    dist[:, torch.arange(node_num), torch.arange(node_num)] = 0.0
    scaled_dist = dist.float() / scaler

    # Dual
    coord_diff_d = node_coords_dual.unsqueeze(2) - node_coords_dual.unsqueeze(1)
    dist_d = torch.norm(coord_diff_d, p=2, dim=-1)
    dist_d[:, torch.arange(node_num), torch.arange(node_num)] = 0.0
    scaled_dist_d = dist_d.float() / scaler

    # ===================== 需求矩阵 =====================
    demand_mat = node_demand.unsqueeze(1).expand(batch_size, node_num, node_num)
    demand_mat_d = node_demand.unsqueeze(1).expand(batch_size, node_num, node_num)

    # ===================== 堆叠 =====================
    problems = torch.stack((scaled_dist, demand_mat), dim=-1)
    dual_problems = torch.stack((scaled_dist_d, demand_mat_d), dim=-1)

    return problems, dual_problems, node_coords, node_demand


def get_dataset_problem(load_path, batch_size):
    filename = load_path
    data = pd.read_csv(filename, sep=',', header=None).to_numpy()

    depot_size = 1
    customer_size = int(data[0][0])
    scale = int(data[0][1])
    capacity = int(data[0][2])

    depot_xy = data[1:depot_size+1] / scale
    pick_xy = data[depot_size+1 : depot_size+1+customer_size]
    delivery_xy = data[depot_size+1+customer_size : depot_size+1+2*customer_size]

    depot_x_y = torch.FloatTensor(depot_xy[:, 0:2]).unsqueeze(0)
    pick_x_y = torch.FloatTensor(pick_xy[:, 0:2]).unsqueeze(0)
    delivery_x_y = torch.FloatTensor(delivery_xy[:, 0:2]).unsqueeze(0)

    pick_demand = torch.FloatTensor(pick_xy[:, 2:3]).unsqueeze(0)
    delivery_demand = torch.FloatTensor(delivery_xy[:, 2:3]).unsqueeze(0)

    # 扩 batch
    depot_x_y = depot_x_y.repeat(batch_size, 1, 1)
    pick_x_y = pick_x_y.repeat(batch_size, 1, 1)
    delivery_x_y = delivery_x_y.repeat(batch_size, 1, 1)
    pick_demand = pick_demand.repeat(batch_size, 1, 1)
    delivery_demand = delivery_demand.repeat(batch_size, 1, 1)

    pick_demand = pick_demand / capacity
    delivery_demand = delivery_demand / capacity

    # 拼接需求 (Env 必须用！)
    depot_demand = torch.zeros(batch_size, 1, 1)
    node_demand = torch.cat([depot_demand, pick_demand, delivery_demand], dim=1).squeeze(-1)
    node_coords = torch.cat([depot_x_y, pick_x_y, delivery_x_y], dim=1)
    B, N, _ = node_coords.shape

    # 距离矩阵
    coord_diff = node_coords.unsqueeze(2) - node_coords.unsqueeze(1)
    dist = torch.norm(coord_diff, p=2, dim=-1)
    dist[:, torch.arange(N), torch.arange(N)] = 0.0
    scaled_dist = dist / scale

    # Dual
    node_coords_d = torch.cat([depot_x_y, delivery_x_y, pick_x_y], dim=1)
    coord_diff_d = node_coords_d.unsqueeze(2) - node_coords_d.unsqueeze(1)
    dist_d = torch.norm(coord_diff_d, p=2, dim=-1)
    dist_d[:, torch.arange(N), torch.arange(N)] = 0.0
    scaled_dist_d = dist_d / scale

    # 需求矩阵
    demand_mat = node_demand.unsqueeze(1).expand(B, N, N)
    demand_mat_d = node_demand.unsqueeze(1).expand(B, N, N)

    # 堆叠
    problems = torch.stack((scaled_dist, demand_mat), dim=-1)
    dual_problems = torch.stack((scaled_dist_d, demand_mat_d), dim=-1)

    # ✅ 恢复输出 node_demand！
    return problems, dual_problems, node_coords, node_demand