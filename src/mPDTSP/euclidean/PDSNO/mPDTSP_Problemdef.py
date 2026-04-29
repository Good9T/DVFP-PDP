import torch

def get_random_problems(batch_size, customer_size, problem_gen_params):
    scaler = problem_gen_params['scaler']
    capacity = problem_gen_params['capacity']
    demand_min = problem_gen_params['demand_min']
    demand_max = problem_gen_params['demand_max']

    depot_size = 1
    node_num = depot_size + customer_size * 2

    depot_xy = torch.rand(batch_size, depot_size, 2)
    pick_xy = torch.rand(batch_size, customer_size, 2)
    delivery_xy = torch.rand(batch_size, customer_size, 2)

    pick_demand = torch.randint(demand_min, demand_max, (batch_size, customer_size))
    delivery_demand = -pick_demand

    pick_demand = pick_demand / capacity
    delivery_demand = delivery_demand / capacity

    depot_demand = torch.zeros(batch_size, 1)
    node_demand = torch.cat([depot_demand, pick_demand, delivery_demand], dim=1)

    node_coords = torch.cat([depot_xy, pick_xy, delivery_xy], dim=1) / scaler
    node_coords_dual = torch.cat([depot_xy, delivery_xy, pick_xy], dim=1) / scaler

    node_demand_expand = node_demand.unsqueeze(-1)
    problems = torch.cat([node_coords, node_demand_expand], dim=-1)
    dual_problems = torch.cat([node_coords_dual, node_demand_expand], dim=-1)

    return problems, dual_problems, node_coords, node_demand