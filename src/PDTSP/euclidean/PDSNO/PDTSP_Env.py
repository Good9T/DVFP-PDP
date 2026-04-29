from dataclasses import dataclass
import torch

# 🔥 调用你的欧式距离问题生成
from PDTSP_Problemdef import get_random_problems


@dataclass
class Reset_State:
    problems: torch.Tensor
    dual_problems: torch.Tensor


@dataclass
class Step_State:
    BATCH_IDX: torch.Tensor
    POMO_IDX: torch.Tensor
    selected_count: int = None
    current_node: torch.Tensor = None
    state_mask: torch.Tensor = None
    finished: torch.Tensor = None


class PDTSPEuclideanEnv:
    def __init__(self, **env_params):
        self.env_params = env_params
        self.pomo_size = env_params['pomo_size']
        self.customer_size = env_params['customer_size']
        self.node_num = 1 + self.customer_size * 2
        self.problem_gen_params = self.env_params['problem_gen_params']

        self.batch_size = None
        self.BATCH_IDX = None
        self.POMO_IDX = None

        self.problems = None       # Primal 坐标 (B, N, 2)
        self.dual_problems = None  # Dual 坐标 (B, N, 2)

        self.selected_count = None
        self.current_node = None
        self.selected_node_list = None
        self.visited_flag = None
        self.mask = None
        self.lock = None
        self.finished = None
        self.step_state = None

    def load_problems(self, batch_size, aug_enable=False, aug_factor=1):
        self.batch_size = batch_size


        self.problems, self.dual_problems = get_random_problems(
            self.batch_size, self.customer_size, self.problem_gen_params
        )
        if aug_enable:
            self.problems = self.problems.repeat(aug_factor, 1, 1)
            self.dual_problems = self.dual_problems.repeat(aug_factor, 1, 1)
            self.batch_size = self.batch_size * aug_factor

        self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.pomo_size)
        self.POMO_IDX = torch.arange(self.pomo_size)[None, :].expand(self.batch_size, self.pomo_size)

    def load_dual_problems(self):
        # 🔥 切换对偶视角（坐标）
        self.problems = self.dual_problems

    def reset(self):
        self.selected_count = 0
        self.current_node = None
        self.selected_node_list = torch.empty((self.batch_size, self.pomo_size, 0), dtype=torch.long)

        self.visited_flag = torch.zeros((self.batch_size, self.pomo_size, self.node_num))
        self.lock = torch.zeros((self.batch_size, self.pomo_size, self.node_num))
        self.mask = torch.zeros((self.batch_size, self.pomo_size, self.node_num))

        self.lock[:, :, 1 + self.customer_size:] = float('-inf')
        self.finished = torch.zeros((self.batch_size, self.pomo_size), dtype=torch.bool)

        self.step_state = Step_State(BATCH_IDX=self.BATCH_IDX, POMO_IDX=self.POMO_IDX)
        return Reset_State(problems=self.problems, dual_problems=self.dual_problems), None, False

    def pre_step(self):
        reward = None
        done = False
        self.step_state.selected_count = self.selected_count
        self.step_state.current_node = self.current_node
        self.step_state.state_mask = self.mask
        self.step_state.finished = self.finished
        return self.step_state, reward, done

    def step(self, selected):
        self.selected_count += 1
        self.current_node = selected
        self.selected_node_list = torch.cat([self.selected_node_list, selected[:, :, None]], dim=2)

        self.visited_flag[self.BATCH_IDX, self.POMO_IDX, selected] = float('-inf')

        is_pick = (selected > 0) & (selected <= self.customer_size)
        unlock = selected.clone()
        unlock[is_pick] += self.customer_size
        self.lock[self.BATCH_IDX, self.POMO_IDX, unlock] = 0

        self.mask = self.visited_flag.clone() + self.lock.clone()

        new_finished = (self.visited_flag == float('-inf')).all(dim=2)
        self.finished = self.finished | new_finished

        self.mask[self.BATCH_IDX, self.POMO_IDX, 0][self.finished] = 0

        # 更新状态
        self.step_state.selected_count = self.selected_count
        self.step_state.current_node = self.current_node
        self.step_state.state_mask = self.mask
        self.step_state.finished = self.finished

        done = self.finished.all()
        reward = -self._get_total_cost() if done else None
        return self.step_state, reward, done

    def _get_total_cost(self):
        index_to_gather = self.selected_node_list[:, :, :, None].expand(-1, -1, -1, 2)
        node_coords = self.problems[:, None, :, :].expand(-1, self.pomo_size, -1, -1)
        seq_ordered = node_coords.gather(dim=2, index=index_to_gather)
        seq_rolled = seq_ordered.roll(dims=2, shifts=-1)
        segment_lengths = ((seq_ordered - seq_rolled) ** 2).sum(3).sqrt()
        travel_distances = segment_lengths.sum(2)
        # shape : (batch, mt)
        return travel_distances