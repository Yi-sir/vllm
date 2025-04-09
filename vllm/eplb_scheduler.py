from typing import Callable, Optional

import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.layer import FusedMoE

logger = init_logger(__name__)


class EPLBScheduler:
    """
    EPLB调度器, 仅在vllm/model_executor/layers/fused_moe/layer.py中使用
    """

    def __init__(
        self,
        num_experts: int,
        topk: int,
        prefix: str = "",
        use_group_topk: bool = False,
        renormalize: bool = True,
        num_expert_group: Optional[int] = None,
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        e_score_correction_bias: Optional[torch.Tensor] = None,
    ):
        self.global_expert_map = torch.arange(num_experts)  # 最初的专家排布是顺序排列的
        self.history_expert_traffic = torch.zeros(num_experts)  # 每个专家的历史流量
        self.current_expert_traffic = torch.zeros(num_experts)  # 每个专家的当前流量
        self.current_traffic_update_times = 0  # 当前流量更新次数

        assert (
            envs.VLLM_EPLB_EXPERT_REBALANCE_THRESHOLD < 1
            and envs.VLLM_EPLB_EXPERT_REBALANCE_THRESHOLD > 0
        )
        self.expert_rebalance_threshold = int(
            self.global_num_experts * envs.VLLM_EPLB_EXPERT_REBALANCE_THRESHOLD
        )  # 专家重平衡阈值，如果变化的专家超过了这个值，就重新load一次

        assert (
            envs.VLLM_EPLB_MOVING_AVG_FACTOR <= 1
            and envs.VLLM_EPLB_MOVING_AVG_FACTOR >= 0
        )
        self.moving_avg_factor = envs.VLLM_EPLB_MOVING_AVG_FACTOR

        assert envs.VLLM_EPLB_TRAFFIC_UPDATE_INTERVAL > 0
        self.traffic_update_interval = (
            envs.VLLM_EPLB_TRAFFIC_UPDATE_INTERVAL
        )  # 流量更新间隔

        assert envs.VLLM_EPLB_NUM_REDUNDANT_EXPERTS >= 0
        self.num_redundant_experts = (
            envs.VLLM_EPLB_NUM_REDUNDANT_EXPERTS
        )  # 冗余专家的数量

        # for debug
        self.layer_idx = prefix.split(sep=".")[-3]  # "idx.mlp.experts"

        self.use_group_topk = use_group_topk
        self.top_k = topk
        self.renormalize = renormalize
        self.num_expert_group = num_expert_group
        self.custom_routing_function = custom_routing_function
        self.scoring_func = scoring_func
        self.e_score_correction_bias = e_score_correction_bias

    def schedule(
        self, hidden_states: torch.Tensor, router_logits: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """
        Expert Parallelism Load Balancing Schedule.
        1. 由输入计算topk专家, 这里的计算其实和moe算子里的冗余了
        2. 统计流量
        3. 每记录self.traffic_update_interval次, 更新一次历史流量

        Args:
            hidden_states: [tokens, hidden_size] 隐藏状态, 其实没用. 计算topk不需要这个参数(vllm的代码只是check shape和取device)
            router_logits: [tokens, num_experts] gate的输出

        Returns:
            history_expert_traffic: [num_experts] 若满足更新条件(启用EP, 当前rank为调度rank, 到达迭代次数)则返回历史流量, 否则返回None
        """
        # 计算topk id，for eplb
        # topk_ids: [tokens, self.top_k]
        _, topk_ids = FusedMoE.select_experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
            use_group_topk=self.use_grouped_topk,
            top_k=self.top_k,
            renormalize=self.renormalize,
            topk_group=self.topk_group,
            num_expert_group=self.num_expert_group,
            custom_routing_function=self.custom_routing_function,
            scoring_func=self.scoring_func,
            e_score_correction_bias=self.e_score_correction_bias,
        )
        # 记录当前流量
        self.current_expert_traffic[topk_ids] += torch.bincount(
            topk_ids.view(-1), minlength=self.global_num_experts
        )
        self.current_traffic_update_times += 1
        if self.current_traffic_update_times % self.traffic_update_interval == 0:
            # 更新历史流量
            # 这里历史流量要不要做一个截断？
            self.history_expert_traffic = (
                self.current_expert_traffic * (1 - self.moving_avg_factor)
                + self.history_expert_traffic * self.moving_avg_factor
            )
            self.current_expert_traffic.zero_()
            self.current_traffic_update_times = 0
            return self.history_expert_traffic
        else:
            return None
