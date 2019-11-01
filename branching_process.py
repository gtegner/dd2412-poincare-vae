import torch
from torch.utils.data import Dataset, DataLoader


from collections import defaultdict
import numpy as np

from sklearn.preprocessing import StandardScaler


class Node:
    def __init__(self, parent=None, name='root', num=0, dim=2, num_x=5, sigma_x=1):
        self.left = None
        self.right = None
        self.parent = parent
        self.name = name
        self.num = num
        self.dim = dim
        self.num_x = num_x
        self.sigma_x = sigma_x

        if self.parent is None:
            self.sampling_mu = np.zeros((self.dim))
        else:
            self.sampling_mu = self.parent.sample_y()

        self.y = None

    def create_children(self, node_num, k=0):

        self.left = Node(parent=self, num=node_num + 1,
                         name=f"left: parent {self.num}, depth={k}, num={node_num + 1}",
                         dim=self.dim,
                         num_x=self.num_x,
                         sigma_x=self.sigma_x)

        self.right = Node(parent=self, num=node_num + 2,
                          name=f"right: parent {self.num}, depth={k}, num={node_num + 2}",
                          dim=self.dim,
                          num_x=self.num_x,
                          sigma_x=self.sigma_x)

        return [self.left, self.right]

    def sample_y(self):
        # Sampling more times samples from the same branching process
        if self.y is None:
            y = self.sampling_mu + np.random.randn((self.dim)) * 1
            self.y = y

        return self.y

    def sample_x(self):

        y_i = self.sample_y()
        x = np.asarray([y_i + np.random.randn(*y_i.shape)*self.sigma_x /
                        np.sqrt(self.num_x) for _ in range(self.num_x)])

        label = np.ones((len(x), 1)) * self.num
        return np.append(x, label, axis=1)

    def __str__(self):
        return self.name


class Tree:
    def __init__(self, max_depth, dim, sigma_x=1, num_x=5, seed=42):
        np.random.seed(seed)

        self.max_depth = max_depth
        self.root = Node(None, dim=dim, sigma_x=sigma_x, num_x=num_x, num=0)

        self.dim = dim
        self.tree_dict = defaultdict(list)
        self.num_nodes = 1
        self.seed = 42

    def generate_tree(self):
        max_depth = self.max_depth
        self.tree_dict[0] = [self.root]
        if max_depth <= 1:
            return
        self.num_nodes = 0
        child1, child2 = self.root.create_children(
            node_num=self.num_nodes, k=1)

        self.tree_dict[1] = [child1, child2]
        self.num_nodes += 2

        if max_depth <= 2:
            return self.tree_dict

        for i in range(2, max_depth):
            for child in self.tree_dict[i-1]:
                ch1, ch2 = child.create_children(node_num=self.num_nodes, k=i)
                self.tree_dict[i].extend([ch1, ch2])
                self.num_nodes += 2

    def sample_x(self):
        if len(self.tree_dict) == 0:
            print("Generate tree first to sample!")
            raise ValueError

        x_samples = []
        for depth, children in self.tree_dict.items():
            for ix, ch in enumerate(children):
                x = ch.sample_x()
                x_samples.append(x)

        x_samples = np.asarray(x_samples)

        return x_samples

    def print_tree(self):
        for k, v in self.tree_dict.items():
            print(f"\n Depth {k}")
            l = [print(ch.__str__(), end=' | ') for ch in v]


class BranchingDataset(Dataset):
    def __init__(self, dim=4, sigma=1, num_x=5, max_depth=3, sigma_x=1, tree=None, train=True, scaler=None, seed=42):

        self.scaler = None
        if scaler is None:
            self.scaler = StandardScaler()
        else:
            self.scaler = scaler

        if tree is None:
            tree = Tree(dim=dim, max_depth=max_depth,
                        sigma_x=sigma_x, num_x=num_x, seed=seed)
            tree.generate_tree()

        dim = tree.dim

        X = tree.sample_x()

        x = X[:, :, 0:-1].reshape(-1, dim)
        y = X[:, :, -1].reshape(-1)
        if train:
            print("Normalizing x...")
            x = self.scaler.fit_transform(x)
        else:
            x = self.scaler.transform(x)

        self.x = x

        self.data = torch.from_numpy(x).float()
        self.y = torch.from_numpy(y).float()

        print("Generated Branching Process, number of samples: ", len(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, ix):
        return self.data[ix], self.y[ix]


class BranchingLoaders:
    def __init__(self, dim, batch_size, test_batch=None, sigma=1, num_x=5, max_depth=6, sigma_x=1, seed=42):
        tree = Tree(dim=dim, max_depth=max_depth,
                    sigma_x=sigma_x, num_x=num_x, seed=seed)
        tree.generate_tree()

        train_dataset = BranchingDataset(tree=tree, train=True)
        test_dataset = BranchingDataset(
            tree=tree, train=False, scaler=train_dataset.scaler)

        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(
            test_dataset, batch_size=test_batch or batch_size, shuffle=True)

    def get_data_loaders(self):
        return self.train_loader, self.test_loader
