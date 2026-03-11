import math
import numpy as np
import torch
import copy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MinMaxStats:
    """Tracks min/max Q-values for dynamic normalization to [0, 1]."""

    def __init__(self):
        self.maximum = -float("inf")
        self.minimum = float("inf")

    def update(self, value):
        self.maximum = max(self.maximum, float(value))
        self.minimum = min(self.minimum, float(value))

    def normalize(self, value):
        if isinstance(value, np.ndarray):
            if self.maximum > self.minimum:
                return (value - self.minimum) / (self.maximum - self.minimum)
            return value
        else:
            if self.maximum > self.minimum:
                return (value - self.minimum) / (self.maximum - self.minimum)
            return value


class Node:
    """MCTS tree node for traffic routing decisions."""

    def __init__(self, action_size, min_max_stats, parent=None, action=None, cpuct=1.1):
        self.parent = parent
        self.action = action
        self.cpuct = cpuct
        self.min_max_stats = min_max_stats
        self.is_expanded = False
        self.children = {}

        self.child_priors = np.zeros(action_size, dtype=np.float32)
        self.child_total_value = np.zeros(action_size, dtype=np.float32)
        self.child_number_visits = np.zeros(action_size, dtype=np.float32)
        self.available_actions = np.ones(action_size, dtype=bool)

    @property
    def number_visits(self):
        if self.parent is None:
            return np.sum(self.child_number_visits)
        return self.parent.child_number_visits[self.action]

    @number_visits.setter
    def number_visits(self, value):
        if self.parent is not None:
            self.parent.child_number_visits[self.action] = value

    @property
    def total_value(self):
        if self.parent is None:
            return 0
        return self.parent.child_total_value[self.action]

    @total_value.setter
    def total_value(self, value):
        if self.parent is not None:
            self.parent.child_total_value[self.action] = value

    def child_Q(self, normalize=False):
        denominator = np.where(self.child_number_visits == 0, 1, self.child_number_visits)
        q = self.child_total_value / denominator
        if normalize:
            q_norm = self.min_max_stats.normalize(q)
            if isinstance(q_norm, np.ndarray):
                q_norm = np.where(q_norm > 0, q_norm, 0)
            return q_norm
        return q

    def child_U(self):
        return (math.sqrt(self.number_visits) *
                self.child_priors / (1 + self.child_number_visits))

    def best_child(self):
        """Select child via UCB: -Q_norm + cpuct * U (lower cost = better)."""
        q = self.child_Q(normalize=True)
        u = self.child_U()
        ucb = -q + self.cpuct * u
        masked_ucb = np.where(self.available_actions, ucb, float('-inf'))
        return int(np.argmax(masked_ucb))

    def select_leaf(self, depth=0, max_depth=5):
        """Traverse tree to find an unexpanded node or depth limit."""
        current = self
        while current.is_expanded and depth < max_depth:
            best_action = current.best_child()
            if best_action not in current.children:
                return current, best_action, depth
            current = current.children[best_action]
            depth += 1
        return current, None, depth

    def expand(self, child_priors, available_actions):
        """Expand node with policy priors."""
        self.is_expanded = True
        self.available_actions = available_actions
        mask = available_actions.astype(np.float32)
        masked_priors = child_priors * mask
        total = masked_priors.sum()
        if total > 0:
            self.child_priors = masked_priors / total
        else:
            n_avail = mask.sum()
            self.child_priors = mask / max(n_avail, 1)

    def backup(self, value):
        """Propagate value estimate up the tree."""
        current = self
        while current.parent is not None:
            current.total_value += value
            current.number_visits += 1
            current.min_max_stats.update(
                current.parent.child_total_value[current.action] /
                max(current.parent.child_number_visits[current.action], 1)
            )
            current = current.parent


class SelectiveMCTS:
    """
    MCTS search adapted for real-time traffic routing.
    Uses selective application: only runs full search when policy is uncertain.
    """

    def __init__(self, model, action_dim, num_simulations=100,
                 max_depth=5, cpuct=1.1, selection_coef=0.75):
        self.model = model
        self.action_dim = action_dim
        self.num_simulations = num_simulations
        self.max_depth = max_depth
        self.cpuct = cpuct
        self.selection_coef = selection_coef

    def should_use_mcts(self, probs):
        """
        Improved uncertainty detection using entropy.
        MCTS is triggered when policy entropy indicates high uncertainty.
        High entropy => uncertain policy => use MCTS for better exploration.
        """
        # Ensure probs is flattened and normalized
        probs = probs.flatten()
        probs = probs / (probs.sum() + 1e-8)
        
        # Compute normalized entropy (0 = certain, 1 = uniform)
        entropy = -np.sum(probs * np.log(probs + 1e-8))
        max_entropy = np.log(len(probs))
        normalized_entropy = entropy / (max_entropy + 1e-8)
        
        # Use MCTS when entropy is above threshold (policy is uncertain)
        # Lower threshold = more MCTS usage
        return normalized_entropy > self.selection_coef

    @torch.no_grad()
    def search(self, state, avail_actions, cur_edge_idx=None):
        """
        Run selective MCTS search.

        Args:
            state: numpy array (flat_state_dim,)
            avail_actions: list/array of 0/1 action mask
            cur_edge_idx: int, index of current edge

        Returns:
            action: int, selected action
            probs: numpy array, action probabilities from policy
            value: float, estimated value
            used_mcts: bool, whether MCTS was actually used
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        mask_tensor = torch.FloatTensor(avail_actions).unsqueeze(0).to(device)
        edge_idx_tensor = None
        if cur_edge_idx is not None:
            edge_idx_tensor = torch.LongTensor([cur_edge_idx]).to(device)

        self.model.eval()
        probs, value = self.model(state_tensor, action_mask=mask_tensor, cur_edge_idx=edge_idx_tensor)
        probs_np = probs.cpu().numpy().squeeze()
        value_np = value.cpu().item()

        avail_mask = np.array(avail_actions, dtype=bool)
        n_available = avail_mask.sum()
        if n_available <= 1:
            action = int(np.argmax(avail_actions))
            return action, probs_np, value_np, False

        if not self.should_use_mcts(probs_np):
            masked_probs = probs_np * avail_mask.astype(np.float32)
            action = int(np.argmax(masked_probs))
            return action, probs_np, value_np, False

        encoding = self.model.get_encoding(state_tensor)
        cav_features = state_tensor[:, self.model.num_edges * self.model.road_feature_dim:]

        min_max_stats = MinMaxStats()
        root = Node(self.action_dim, min_max_stats, cpuct=self.cpuct)
        root.expand(probs_np, avail_mask)

        for _ in range(self.num_simulations):
            node, unexpanded_action, depth = root.select_leaf(max_depth=self.max_depth)

            if unexpanded_action is not None:
                child = Node(self.action_dim, min_max_stats, parent=node,
                             action=unexpanded_action, cpuct=self.cpuct)
                node.children[unexpanded_action] = child

                child_probs, child_value = self.model.decode_from_encoding(
                    encoding, cav_features, cur_edge_idx=edge_idx_tensor,
                    action_mask=mask_tensor
                )
                child_probs_np = child_probs.cpu().numpy().squeeze()
                child_value_np = child_value.cpu().item()

                child.expand(child_probs_np, avail_mask)
                child.backup(child_value_np)
            else:
                _, leaf_value = self.model.decode_from_encoding(
                    encoding, cav_features, cur_edge_idx=edge_idx_tensor,
                    action_mask=mask_tensor
                )
                node.backup(leaf_value.cpu().item())

        visit_counts = root.child_number_visits
        masked_visits = visit_counts * avail_mask.astype(np.float32)
        total_visits = masked_visits.sum()
        if total_visits > 0:
            action_probs = masked_visits / total_visits
        else:
            action_probs = avail_mask.astype(np.float32) / max(n_available, 1)

        action = int(np.argmax(action_probs))
        return action, action_probs, value_np, True
