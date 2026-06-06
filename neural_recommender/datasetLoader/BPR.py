import numpy as np
from .BaseDataset import BaseDataset


class BPRDataset(BaseDataset):
    def __init__(self, config, interactions: np.ndarray):
        super().__init__(config, interactions)
        # define params
        self.interactions = interactions
        self.neg_samples = config.neg_samples
        self.rng = np.random.default_rng(config.seed if config.seed is not None else 42)
        #
        self.user_pos = {u: np.where(interactions[u] > 0)[0] for u in range(interactions.shape[0])}
        self.all_items = np.arange(interactions.shape[1])
        # one sample per positive interaction
        self.samples = [(u, i) for u in self.user_pos for i in self.user_pos[u]]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        u, i_pos = self.samples[idx]
        # multi negative sampling
        negs = []
        while len(negs) < self.neg_samples:
            n = self.rng.choice(self.all_items)
            if n not in self.user_pos[u]:
                negs.append(n)
        return np.int64(u), np.int64(i_pos), np.array(negs, dtype=np.int64)
