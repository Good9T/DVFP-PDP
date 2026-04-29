import torch

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
    node_coords = torch.cat([depot_xy, pick_xy, delivery_xy], dim=1) / scaler

    # ======================
    # Dual 对偶视角：取送交换 [depot, D, P]
    # ======================
    node_coords_dual = torch.cat([depot_xy, delivery_xy, pick_xy], dim=1) / scaler

    problems = node_coords
    dual_problems = node_coords_dual

    return problems, dual_problems


