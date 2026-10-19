"""
lr_policy.py — LeRobot で学習した方策（ACT など）を、今回のシミュレーションの実行ループ（bc_policy.rollout）から使うための包み
  pol = LRPolicy("outputs/act_kit/checkpoints/last/pretrained_model")
  chunk = pol.act(obs14)   # (10, 8)：1 秒先までの関節目標 7 と把持指令
実行のしかた（10 Hz、重なった予測の加重平均）は自作の方策と同じ bc_policy.rollout を使うので、ネットワークだけを比べられる
"""
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class, make_pre_post_processors


class LRPolicy:
    def __init__(self, path, device="cpu"):
        cfg = PreTrainedConfig.from_pretrained(path); cfg.device = device
        self.policy = get_policy_class(cfg.type).from_pretrained(path, config=cfg)
        self.policy.eval()
        self.pre, self.post = make_pre_post_processors(cfg, pretrained_path=path)

    @torch.no_grad()
    def act(self, obs):
        o = torch.as_tensor(obs, dtype=torch.float32)[None]
        batch = self.pre({"observation.state": o[:, :9], "observation.environment_state": o[:, 9:]})
        chunk = self.policy.predict_action_chunk(batch)          # (1, K, 8)（正規化された値）
        chunk = self.post(chunk)                                  # 元の単位へ戻す
        return chunk[0].cpu().numpy()
