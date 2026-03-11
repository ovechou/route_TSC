import torch

from agent import CAVAgent
from algo.alpha_router_net import AlphaRouterModel
from algo.alpha_router_mcts import SelectiveMCTS
from algo.alpha_router_buffer import AlphaRouterTrainer
import config

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class AlphaRouterWrapper:
    """
    Wrapper that exposes the same interface as DQNAgent/PPOAgent
    so it can be used as the shared `router` object in episode.run().

    Training uses REINFORCE + value baseline (aligned with original AlphaRouter).
    MCTS is only used during evaluation (execute=True).
    """

    def __init__(self, state_dim, action_dim, args, ar_config=None):
        if ar_config is None:
            ar_config = config.ALPHA_ROUTER

        self.args = args
        self.action_dim = action_dim
        self.ar_config = ar_config

        num_edges = args.agent_num * 4 * 3
        embedding_dim = ar_config["EMBEDDING_DIM"]
        encoder_layers = ar_config["ENCODER_LAYERS"]
        num_heads = ar_config["NUM_HEADS"]
        qkv_dim = ar_config["QKV_DIM"]
        C = ar_config.get("C", 10)

        self.model = AlphaRouterModel(
            road_feature_dim=args.road_feature,
            cav_feature_dim=args.cav_feature,
            action_dim=action_dim,
            embedding_dim=embedding_dim,
            encoder_layers=encoder_layers,
            num_heads=num_heads,
            qkv_dim=qkv_dim,
            num_edges=num_edges,
            C=C,
        ).to(device)

        self.trainer = AlphaRouterTrainer(
            model=self.model,
            lr=ar_config["LR"],
            gamma=ar_config.get("GAMMA", 0.99),
            val_loss_coef=ar_config["VAL_LOSS_COEF"],
            ent_coef=ar_config["ENT_COEF"],
            max_grad_norm=ar_config.get("MAX_GRAD_NORM", 1.0),
        )

        self.mcts = SelectiveMCTS(
            model=self.model,
            action_dim=action_dim,
            num_simulations=ar_config["NUM_SIMULATIONS"],
            max_depth=ar_config["MAX_DEPTH"],
            cpuct=ar_config["CPUCT"],
            selection_coef=ar_config["SELECTION_COEF"],
        )

        self.use_mcts = ar_config.get("USE_MCTS", True)
        self.buf_cnt = 0
        self.training = False
        self.scheduler = None

    def act(self, state, qnet=None, avail_actions=None, execute=False):
        """
        Training: pure policy sampling (exploration via entropy bonus).
        Evaluation: MCTS or greedy argmax.
        Returns: (action, log_prob, value)
        """
        if avail_actions is None:
            avail_actions = [1] * self.action_dim

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        mask_tensor = torch.FloatTensor(avail_actions).unsqueeze(0).to(device)

        if execute and self.use_mcts:
            action, probs_np, value_np, used_mcts = self.mcts.search(
                state, avail_actions
            )
            probs_t = torch.FloatTensor(probs_np).to(device).clamp(min=1e-8)
            probs_t = probs_t / probs_t.sum()
            dist = torch.distributions.Categorical(probs_t)
            log_prob = dist.log_prob(torch.tensor(action).to(device)).item()
            return action, log_prob, value_np

        self.model.eval()
        with torch.no_grad():
            probs, value = self.model(state_tensor, action_mask=mask_tensor)

        dist = torch.distributions.Categorical(probs)

        if execute:
            action = probs.argmax(dim=-1).squeeze().item()
        else:
            action = dist.sample().item()

        log_prob = dist.log_prob(torch.tensor(action).to(device)).item()
        value_scalar = value.squeeze().item()

        return action, log_prob, value_scalar

    def new_trajectory(self, cav_id):
        """Register a new CAV trajectory in the buffer."""
        self.trainer.new_trajectory(cav_id)

    def store_step(self, cav_id, state, action, log_prob, value, action_mask=None):
        """Store one step for a specific CAV trajectory."""
        self.buf_cnt += 1
        self.trainer.store_step(cav_id, state, action, log_prob, value, action_mask)

    def finish_trajectory(self, cav_id, final_reward):
        """Mark a CAV trajectory as complete with its final reward."""
        self.trainer.finish_trajectory(cav_id, final_reward)

    def discard_trajectory(self, cav_id):
        """Discard an incomplete CAV trajectory."""
        self.trainer.discard_trajectory(cav_id)

    def store(self, state, reward, next_state, done, q=None, actions=None):
        """Legacy interface — not used by AlphaRouter, kept for compatibility."""
        pass

    def learn(self):
        """Train once per episode using all completed trajectories."""
        if self.trainer.buffer.num_trajectories < 1:
            return False
        self.training = True
        loss = self.trainer.learn()
        return loss

    def save_model(self, name):
        self.trainer.save_model(f'./model/{name}/alpha_router.pt')

    def load_model(self, name):
        self.trainer.load_model(f'./model/{name}/alpha_router.pt')


class AlphaRouterCAVAgent(CAVAgent):
    """
    CAVAgent subclass for AlphaRouter algorithm.
    Manages per-CAV trajectory accumulation and communicates
    with the shared AlphaRouterWrapper (router).
    """

    def __init__(self, veh_id, router, adj_edge, args):
        super(AlphaRouterCAVAgent, self).__init__(veh_id, router, adj_edge, args)
        self.traj_registered = False
        self._last_log_prob = 0.0
        self._last_value = 0.0
        self._last_action_mask = None

    def _ensure_traj(self):
        if not self.traj_registered:
            self.router.new_trajectory(self.veh_id)
            self.traj_registered = True
