#!/usr/bin/python
"""This is a short description.
Replace this with a more detailed description of what this file contains.
"""
import time

import argparse
import os.path as osp
import os
import json
from collections import Counter, defaultdict
import pickle

import numpy as np
from datetime import datetime

from tqdm import tqdm
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, dataset

import knockoff.config as meconfig
# import modelextract.config as meconfig # knockoff/config.py?

# from modelextract.substitutemodel.active.active import get_transfer_dataset
from knockoff import datasets
from knockoff.utils import model as model_utils, transforms

__author__ = "Tribhuvanesh Orekondy"
__maintainer__ = "Tribhuvanesh Orekondy"
__email__ = "orekondy@mpi-inf.mpg.de"
__status__ = "Development"

valid_rewards = ['margin', 'marginadv', 'margincomb', 'diversity', 'improv', 'imitate']
MIN_NODE_IMAGES = 4


class BanditNode:
    def __init__(self, parent, modename, dataset, hierarchy, depth=0, alpha=None, parent_action_id=-1,
                 use_baseline=True, temp=1., eps=0.0, expansion_visits=0):
        """
        Node
        :param dataset: a pytorch Dataset object
        :param hierarchy: A list of dict elements. Each dict element = {'LabelName': 'xxxx', ('Subcategory': element)}
        """
        self.nodeid = utils.id_generator() # TODO: replace with some system library
        self.parent = parent
        self.modename = modename
        self.is_valid = True
        self.is_leaf = False
        self.children = []
        self.parent_action_id = parent_action_id
        self.alpha = alpha
        self.use_baseline = use_baseline
        self.depth = depth
        self.temp = temp
        self.eps = eps
        self.n_visited = 0  # No. of times this action was sampled
        self.expansion_visits = expansion_visits

        # --------- Tree construction
        # Create children
        for entity in hierarchy:
            child_label = entity['LabelName']
            if 'Subcategory' in entity:
                # Child is a node
                child_node = BanditNode(self, child_label, dataset, entity['Subcategory'], depth=self.depth + 1,
                                        alpha=self.alpha, expansion_visits=self.expansion_visits)
            else:
                # Child is a leaf
                child_node = BanditLeaf(self, child_label, dataset, depth=self.depth + 1)

            if child_node.is_valid:
                self.children.append(child_node)

        # Replicate oneself as leaf
        # Why? Sometimes the parent node also has images attributed to itself
        self_child_node = BanditLeaf(self, self.modename, dataset)
        if self_child_node.is_valid:
            self.children.append(self_child_node)

        # --------- Check if this is a valid node
        # If it doesn't have any valid children, tell parent to drop this node
        if len(self.children) == 0:
            self.is_valid = False

        if not self.is_valid:
            return

        # --------- Define actions, rewards, etc. for this node
        self.nactions = len(self.children)
        self.actions = np.arange(len(self.children))
        self.alpha = alpha
        self.H = np.ones(self.nactions)
        self.pi = np.ones(self.nactions) / self.nactions
        self.N = np.zeros(self.nactions)

        # For debugging
        self.action_to_reward_hist = defaultdict(list)
        self.choices = []  # 0 for exploit, 1 for explore (random action)
        self.actions_hist = []
        self.rewards_hist = []

        for i, c in enumerate(self.children):
            c.parent_action_id = i

    def enable_uniform_prob_leaves(self):
        """
        By default, leaves at lower depths have a higher prob. of being sampled.
        This sets policy of sampling a node proportional to # leaves of that node
        :return:
        """
        leaf_counts = np.array([l.count_leaves() for l in self.children])
        self.pi = leaf_counts / np.sum(leaf_counts)
        self.H = np.log(self.pi)
        for c in self.children:
            if not c.is_leaf:
                c.enable_uniform_prob_leaves()

    def sample_action(self):
        self.n_visited += 1
        if np.random.random() <= self.eps:
            action = np.random.choice(self.actions)
        else:
            action = np.random.choice(self.actions, p=self.pi)

        if self.n_visited >= self.expansion_visits:
            action_node = self.children[action]
            return action_node.sample_action()
        else:
            return self, self.modename

    def random_action(self):
        action = np.random.choice(self.actions)
        action_node = self.children[action]
        return action_node.random_action()

    def populate_leaves(self, lst):
        for child in self.children:
            child.populate_leaves(lst)

    def populate_leaves_references(self, lst):
        for child in self.children:
            child.populate_leaves_references(lst)

    def count_nodes(self):
        return 1 + sum([n.count_nodes() for n in self.children])

    def count_leaves(self):
        return sum([n.count_leaves() for n in self.children])

    def count_images(self):
        return sum([n.count_images() for n in self.children])

    def add_action_reward_pair(self, action, reward):
        self.action_to_reward_hist[action].append(reward)
        self.actions_hist.append(action)
        self.rewards_hist.append(reward)

        self.N[action] += 1
        if self.alpha is None:
            alpha = (1. / self.N[action])
        else:
            alpha = self.alpha

        reward_bar = np.mean(self.rewards_hist) if self.use_baseline else 0.
        for a in self.actions:
            if a == action:
                self.H[a] = self.H[a] + alpha * (reward - reward_bar) * (1 - self.pi[a])
            else:
                self.H[a] = self.H[a] - alpha * (reward - reward_bar) * self.pi[a]
        self.pi = np.exp(self.H / self.temp) / np.sum(np.exp(self.H / self.temp))

        # Ancestors update their rewards
        if self.parent is not None:  # Only none in case of root
            self.parent.add_action_reward_pair(self.parent_action_id, reward)

    def add_reward(self, reward):
        if self.parent is not None:
            self.parent.add_action_reward_pair(self.parent_action_id, reward)

    def __str__(self):
        str_rep = 'Mode = {}\t#Children = {}\tpi={}'.format(self.modename, len(self.children), self.pi)
        return str_rep

    def delete(self):
        idx = self.parent_action_id
        mask = np.ones(len(self.parent.children), dtype=bool)
        mask[idx] = False

        del self.parent.children[idx]
        self.parent.nactions = len(self.parent.children)
        self.parent.actions = np.arange(len(self.parent.children))
        self.parent.H = self.parent.H[mask]
        self.parent.pi = self.parent.pi[mask]
        self.parent.pi /= np.sum(self.parent.pi)
        self.parent.N = self.parent.N[mask]

        for i, c in enumerate(self.parent.children):
            c.parent_action_id = i

        if len(self.parent.children) == 0:
            self.parent.delete()


class BanditLeaf:
    """
    Selecting this leaf is equivalent to performing an action.
    """

    def __init__(self, parent, modename, dataset, depth=0):
        self.nodeid = utils.id_generator() # TODO: replace with some standard library
        self.parent = parent
        self.modename = modename
        self.dataset = dataset
        self.is_valid = True
        self.is_leaf = True
        self.nimages = -1
        self.parent_action_id = -1
        self.depth = depth
        self.n_visited = 0  # No. of times this action was sampled

        # --------- Check if this is a valid node
        # Dataset contains this mode
        if self.modename not in self.dataset.classes:
            self.is_valid = False
            return

        # Has a certain number of images
        modeid = self.dataset.class_to_idx[self.modename]
        self.nimages = len(self.dataset.modeid_to_idx[modeid])
        if self.nimages < MIN_NODE_IMAGES:
            print('Skipping "{}" - only {} (< {}) images found'.format(self.modename, self.nimages, MIN_NODE_IMAGES))
            self.is_valid = False
            return

    def delete(self):
        idx = self.parent_action_id
        mask = np.ones(len(self.parent.children), dtype=bool)
        mask[idx] = False

        del self.parent.children[idx]
        self.parent.nactions = len(self.parent.children)
        self.parent.actions = np.arange(len(self.parent.children))
        self.parent.H = self.parent.H[mask]
        self.parent.pi = self.parent.pi[mask]
        self.parent.pi /= np.sum(self.parent.pi)
        self.parent.N = self.parent.N[mask]

        for i, c in enumerate(self.parent.children):
            c.parent_action_id = i

        if len(self.parent.children) == 0:
            self.parent.delete()

    def sample_action(self):
        self.n_visited += 1
        return self, self.modename

    def random_action(self):
        return self, self.modename

    def populate_leaves(self, lst):
        lst.append(self.modename)

    def populate_leaves_references(self, lst):
        lst.append(self)

    def count_nodes(self):
        return 1

    def count_leaves(self):
        return 1

    def count_images(self):
        return self.nimages

    def add_reward(self, reward):
        self.parent.add_action_reward_pair(self.parent_action_id, reward)

    def __str__(self):
        str_rep = 'Mode = {}, Action = {}'.format(self.modename, self.parent_action_id)
        return str_rep


def main():
    parser = argparse.ArgumentParser(description='Train a model via distillation')
    parser.add_argument('target_model_dir', metavar='PATH', type=str, help='Path to target model')
    parser.add_argument('--out_dir', metavar='PATH', type=str,
                        help='Destination Path to store {data, models, ...}', required=True)
    parser.add_argument('--n_examples', metavar='N', type=int, help='# Examples to use to train student model',
                        required=True)
    parser.add_argument('--teacher_ds', metavar='TYPE', type=str, help='Teacher dataset',
                        required=True)
    parser.add_argument('--student_ds', metavar='TYPE', type=str, help='Student dataset', required=True)
    parser.add_argument('--smodel_arch', metavar='TYPE', type=str,
                        help='Substitute model architectures', required=True)
    parser.add_argument('--loss', metavar='LOSS', type=str, help='Loss',
                        default='sce', choices=['kl_div', 'ce', 'mse', 'bce', 'sce'])
    parser.add_argument('--loss-weight', metavar='N', type=float, help='Loss Weight',
                        default=1.0)
    parser.add_argument('--n_student_test', metavar='N', type=int, help='# student test examples',
                        default=2000)
    parser.add_argument('--topk', metavar='N', type=int, help='Use posteriors only from topk classes',
                        default=None)
    parser.add_argument('--rounding', metavar='N', type=int, help='Round posteriors to these many decimals',
                        default=None)
    parser.add_argument('--use_stud_classes', metavar='CLS', type=str,
                        help='Use only these classes from student dataset', required=False, default=None)
    parser.add_argument('--teacher_net', metavar='TYPE', type=str, help='Teacher Network', required=False, default=None)
    # ----------- Params for PHASE 1: Initialization
    parser.add_argument('--init_batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--n_init_epochs', metavar='N', type=int,
                        help='Use these many examples to initialize the model',
                        default=meconfig.N_INIT_EPOCHS, required=False)
    parser.add_argument('--init_lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)', required=False)
    parser.add_argument('--n_init_examples', metavar='N', type=int,
                        help='Use these many examples to initialize the model',
                        default=0, required=True)
    # ----------- Params for PHASE 2: Online learning
    parser.add_argument('-t', '--hierarchy', type=str, metavar='FILE',
                        help='File containing hierarchy of modes', required=True)
    parser.add_argument('--online_batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--online_lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)', required=False)
    parser.add_argument('--log_interval', metavar='N', type=int, help='Logging interval',
                        default=500)
    parser.add_argument('--eps', metavar='N', type=float, help='Epsilon for eps-greedy policy',
                        default=0.5)
    parser.add_argument('--retrain', action='store_true', help='Retrain student model at specified checkpoints',
                        default=False)
    parser.add_argument('--rewards', metavar='Type', type=str, help='Comma-separated rewards', required=True)
    parser.add_argument('--bandit_alpha', metavar='N', type=float, help='Step-size for bandit updates', default=None)
    parser.add_argument('--baseline_delta', metavar='N', type=int, help='Baseline delta', default=20)
    parser.add_argument('--topk_delta', metavar='N', type=int, help='Baseline delta', default=25)
    parser.add_argument('--weighted_imitate', action='store_true', help='Use weighted imitation loss', default=False)
    parser.add_argument('--random', action='store_true', help='Screw strategies. Randomly sample them all',
                        default=False)
    parser.add_argument('--temp', metavar='N', type=float, help='Use posteriors only from topk classes',
                        default=1.0)
    parser.add_argument('--bandit_eps', metavar='N', type=float, help='Bandit epsilon',
                        default=0.0)
    parser.add_argument('--global_eps', metavar='N', type=float, help='Global epsilon',
                        default=0.0)
    parser.add_argument('--expansion', metavar='N', type=int, help='Expansion criteria',
                        default=0)
    parser.add_argument('--tau_data', metavar='N', type=float, help='Frac. of data to sample from Student data',
                        default=1.0)
    parser.add_argument('--tau_classes', metavar='N', type=float, help='Frac. of classes to sample from Student data',
                        default=1.0)
    # ----------- Other params
    parser.add_argument('-d', '--device', metavar='D', type=int, help='Device id', default=0)
    parser.add_argument('--default_batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('-w', '--nworkers', metavar='N', type=int, help='# Worker threads to load data', default=15)
    parser.add_argument('--tag', metavar='STR', type=str, help='Tag to identify experiment', default='')
    args = parser.parse_args()
    params = vars(args)

    out_dir = params['out_dir']
    print('Files and data will be written to: ', out_dir)
    os.makedirs(out_dir, exist_ok=True)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(params['device'])
    device = torch.device('cuda:0')
    torch.manual_seed(meconfig.DEFAULT_SEED)
    num_workers = params['nworkers']

    expt_tag = params['tag']

    # --------------- Initialize model
    target_model_dir = params['target_model_dir']
    default_batch_size = params['default_batch_size']

    # --------------- Load Test Data
    teacher_ds = params['teacher_ds']
    teacher_net = teacher_ds if params['teacher_net'] is None else params['teacher_net']

    print('Using teacher dataset: ', teacher_ds)
    valid_datasets = datasets.__dict__.keys()
    if teacher_ds not in valid_datasets:
        raise ValueError('Teacher dataset not found. Valid arguments = {}'.format(valid_datasets))
    test_transform = datasets.modelfamily_to_transforms[teacher_net]['test']
    teacher_test_data = datasets.__dict__[teacher_ds](train=False, transform=test_transform)
    teacher_test_loader = torch.utils.data.DataLoader(teacher_test_data, batch_size=default_batch_size, shuffle=False,
                                                      num_workers=num_workers)
    target_net = model_utils.get_net(teacher_net)
    criterion_teacher = nn.CrossEntropyLoss()

    # --------------- Prepare hyperparameters, loggers, etc.
    nclasses = target_net.get_output_shape()[-1]

    student_ds = params['student_ds']
    if student_ds not in valid_datasets:
        raise ValueError('Student dataset not found. Valid arguments = {}'.format(valid_datasets))

    query_budget = params['n_examples']
    student_model_arch = params['smodel_arch']
    temp = 1
    expt_id = '{}'.format(int(time.time()))
    if expt_tag:
        expt_id += '-' + expt_tag

    # Store arguments
    params['completed'] = False
    params['expt_start_dt'] = datetime.now()
    params_out_path = osp.join(out_dir, 'params-active-{}.json'.format(expt_id))
    with open(params_out_path, 'w') as jf:
        json.dump(params, jf, indent=True, default=str)

    criterion_student_str = params['loss']

    np.random.seed(meconfig.DEFAULT_SEED)
    torch.manual_seed(meconfig.DEFAULT_SEED)
    torch.cuda.manual_seed(meconfig.DEFAULT_SEED)

    # --------------- Set up datasets
    student_train_partition, student_test_partition = ('train', 'test')
    print('Using partitions {} and {} on student dataset {}'.format(student_train_partition,
                                                                    student_test_partition,
                                                                    student_ds))
    n_student_train = params['n_examples'] + params['n_init_examples']
    n_student_test = params['n_student_test']
    stud_ds_kwargs = dict()
    # This should be a large dataset with many no. of classes e.g., ImageNet, OpenImages
    # Hierarchy over these classes will be loaded later
    unlabeled_student_train_data = datasets.__dict__[student_ds](train=True, transform=transforms.DefaultTransforms)
    unlabeled_student_test_data = datasets.__dict__[student_ds](train=False, transform=transforms.DefaultTransforms)

    n_student_train = len(unlabeled_student_train_data)
    n_student_test = len(unlabeled_student_test_data)
    print('Size of student TRAIN dataset = ', n_student_train)
    print('Size of student TEST dataset = ', n_student_test)

    tau_data = params['tau_data']
    if tau_data < 1.0:
        print('Sampling {} for dataset'.format(tau_data))
        data_utils.sample_data_(unlabeled_student_train_data, tau_data) # TODO: replace with standard library function (pytorch or numpy)
        n_student_train = len(unlabeled_student_train_data)
        print('NEW Size of student TRAIN dataset = ', n_student_train)

    # import ipdb;
    # ipdb.set_trace()

    # These datasets either have targets = (a) valid classes (b) useless classes
    # Right now, assume they are valid classes/clusters/groups
    for ds in [unlabeled_student_train_data, unlabeled_student_test_data]:
        ds.modes = ds.classes
        ds.mode_ids = [x[1] for x in ds.samples]
        ds.modeid_to_idx = defaultdict(list)
        for idx, cid in enumerate(ds.mode_ids):
            ds.modeid_to_idx[cid].append(idx)
    modeidx_to_modename = unlabeled_student_train_data.modes
    modename_to_modeidx = {v: k for k, v in enumerate(modeidx_to_modename)}

    # with open(meconfig.IMAGENET_CLASSES, 'r') as f:
    #     # Source: https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a
    #     imagenet_class_names = eval(f.read())

    # --------------- Initialize transfer dataset
    # For every input query made to the teacher, append this input + teacher's output probabilities to this dataset
    # Use these examples to train the student
    # TODO: find a solution
    labeled_student_train_data = data_utils.ImageTransferDataset(dataset_name='Student Train Transfer',
                                                                 transform=transforms.DefaultTransforms)

    print('Getting predictions on Student-Test ({} examples)'.format(len(unlabeled_student_test_data)))
    # get predictions for all samples in unlabeled student set
    # TODO: replace with code from transfer.py
    labeled_student_test_data = get_transfer_dataset(target_net, unlabeled_student_test_data, device,
                                                     batch_size=default_batch_size, num_workers=num_workers,
                                                     transfer_probs=True)
    # this is only used for evaluation, so it's fine
    labeled_student_test_loader = DataLoader(labeled_student_test_data, shuffle=False, batch_size=default_batch_size,
                                             num_workers=num_workers)

    # --------------- Set up student network
    model_arch_kwargs = {'pretrained': True}
    # TODO: get code from adverary/train.py
    student_net = model_utils.get_net(student_model_arch, n_output_classes=nclasses,
                                      **model_arch_kwargs)
    student_net.to(device)

    n_init_examples = params['n_init_examples']
    # --------------- Phase 1: Initialization ------------------------------------------------------------
    if n_init_examples > 0:
        batch_size = params['init_batch_size']
        print('Performing initialization using {} examples'.format(n_init_examples))

        unlabeled_student_init_data = dataset.__dict__[student_ds](train=True, transform=transforms.DefaultTransforms)
        # TODO: replace with transfer.py code
        labeled_student_init_data = get_transfer_dataset(target_net, unlabeled_student_init_data, device,
                                                         batch_size=batch_size, num_workers=num_workers,
                                                         transfer_probs=True)
        labeled_student_init_loader = DataLoader(labeled_student_init_data, shuffle=True, batch_size=batch_size,
                                                 num_workers=num_workers)
        init_optimizer = optim.SGD(student_net.parameters(), lr=params['init_lr'], momentum=args.momentum,
                                   weight_decay=5e-4)
        for epoch in range(params['n_init_epochs']):
            student_train_loss, student_train_acc = \
                model_utils.train_step(student_net, labeled_student_init_loader,
                                          criterion_student_str, init_optimizer, epoch, device)
        # Eval on Teacher-Test set
        teacher_test_loss, teacher_test_acc = model_utils.test_step(student_net, teacher_test_loader,
                                                               criterion_teacher, device,
                                                               epoch=params['n_init_epochs'])

        # Copy predictions to student data
        for i in range(len(labeled_student_init_data)):
            labeled_student_train_data.append(labeled_student_init_data.samples[i])

        print()

    # --------------- Phase 2: Adaptive Learning ------------------------------------------------------
    time_start = time.time()
    time_prev = time.time()
    train_losses = []
    log_interval = params['log_interval']
    best_teacher_test_acc = 0.0
    best_student_test_loss = 1e10

    # ---- Optimizer
    batch_size = params['online_batch_size']

    online_lr = params['online_lr']
    do_retrain = params['retrain']
    retrain_lr = 0.01
    # online_optimizer = optim.Adagrad(student_net.parameters(), lr=online_lr)
    online_optimizer = optim.SGD(student_net.parameters(), lr=online_lr, momentum=args.momentum)
    retrain_optimizer = None
    # Right now, using a manual schedule. Think later to fix this depending on loss/acc(student-test)
    retrain_budgets = np.arange(1000, 10000, 1000).tolist()  # Every 1k queries \in [1k, 10k]
    retrain_budgets += np.arange(10000, 30000, 3000).tolist()  # Every 3k queries \in [10k, 30k]
    retrain_budgets += np.arange(30000, 50000, 10000).tolist()  # Every 10k queries \in [30k, 50k]
    retrain_budgets += np.arange(50000, query_budget + 9999, 25000).tolist()  # Every 25k queries \in [50k, B]
    retrain_checkpoints = np.array(retrain_budgets) / batch_size
    retrain_checkpoints = retrain_checkpoints.astype(int)
    if do_retrain:
        # retrain_optimizer = optim.Adagrad(student_net.parameters(), lr=retrain_lr)
        retrain_optimizer = optim.SGD(student_net.parameters(), lr=retrain_lr, momentum=args.momentum)
        print('Will retrain model at: ', retrain_budgets)
    retrain_epochs = 30

    # ---- Logging
    ''' Three files should be written:
    1. Student-train (obtained using some selection policy)
    2. Student-test (predictions of F(x) on held-out set of student images)
    3. Metrics CSV: tracking some metrics during active learning 
    '''
    # 1. This will be flushed every log_interval epochs
    student_train_out_path = osp.join(out_dir, 'samples_student_train.pickle')
    # 2,3
    student_test_out_path = osp.join(out_dir, 'samples_student_test.pickle')
    labeled_student_test_data.dump(student_test_out_path)
    df_out_path = osp.join(out_dir, 'results-online-{}.csv'.format(expt_id))
    df = pd.DataFrame(columns=['time', 'B', 'student_train_loss',
                               'student_test_loss', 'student_test_acc',
                               'teacher_test_loss', 'teacher_test_acc', 'teacher_test_acc_per_class',
                               'a_t', 'r_t'])
    bandit_out_path = osp.join(out_dir, 'bandit_hist.pickle')

    '''
    For each "action" (class, cluster, etc.) in the student dataset, compute a reward per action as follows:
        a. Sample action a using some bandit algorithm
        b. Sample an image batch X_a from distribution 'a'
        c. Get predictions F(X_a)
        d. Take one SGD step using {X_a, F(X_a)}
        e. Compute reward(a)
        f. Update training set
    '''
    # ---- Set up Bandit
    bandit_alpha = params['bandit_alpha']
    hierarchy_path = params['hierarchy']
    temp = params['temp']
    bandit_eps = params['bandit_eps']
    global_eps = params['global_eps']
    expansion_visits = params['expansion']
    with open(hierarchy_path) as rf:
        hierarchy = json.load(rf)
    print('Initializing Hierarchical Bandits')
    bandit_agent = BanditNode(None, hierarchy['LabelName'], unlabeled_student_train_data, hierarchy['Subcategory'],
                              alpha=bandit_alpha, use_baseline=False, temp=temp, eps=bandit_eps,
                              expansion_visits=expansion_visits)
    bandit_agent.enable_uniform_prob_leaves()
    bandit_agent.n_visited = expansion_visits  # No expansion criteria for root node
    print('# Images = {}\t# Actions (leaves) = {}'.format(bandit_agent.count_images(), bandit_agent.count_leaves()))
    # import ipdb; ipdb.set_trace()

    all_idxs = set(range(len(unlabeled_student_train_data)))

    if params['tau_classes'] < 1.0:
        print('Retaining {}% of classes'.format(params['tau_classes'] * 100.))

        # Delete leaves
        leaves = []
        bandit_agent.populate_leaves_references(leaves)
        new_n_leaves = int(params['tau_classes'] * len(leaves))
        leaves_to_delete = np.random.choice(leaves, size=len(leaves) - new_n_leaves, replace=False)
        for l in leaves_to_delete:
            l.delete()
        print('[NEW] # Images = {}\t# Actions (leaves) = {}'.format(bandit_agent.count_images(),
                                                                    bandit_agent.count_leaves()))

        # Re-index remaining images
        # Random sampling access all_idxs and not using bandit agent
        modenames = []
        all_idxs = []
        bandit_agent.populate_leaves(modenames)

        for modename in modenames:
            modeid = modename_to_modeidx[modename]
            if hasattr(unlabeled_student_train_data, 'class_names'):
                modename = unlabeled_student_train_data.class_names[modeid]
            else:
                modename = unlabeled_student_train_data.classes[modeid]
            all_idxs += unlabeled_student_train_data.modeid_to_idx[modeid]

        all_idxs = set(all_idxs)
        print('# Images in index = ', len(all_idxs))

    # ---- Set up Reward Tracking
    reward_names = params['rewards'].split(',')
    n_rewards = len(reward_names)
    rewards_hist_mat = np.zeros((0, n_rewards))  # matrix of size:  T x n_rewards
    delta_baseline = params['baseline_delta']
    assert all([(r in valid_rewards) for r in reward_names])

    y_t_probs_hist = np.zeros((0, nclasses))
    improv_loss_before = 0.0

    mode_counter = Counter()

    start_B = n_init_examples
    end_B = query_budget + n_init_examples
    topk = params['topk']
    rounding = params['rounding']

    action_history = []

    for t, B in enumerate(range(start_B, end_B, batch_size)):
        # ------------------ a. Sample action
        if np.random.random() <= global_eps:
            action_node, a_t = bandit_agent.random_action()
        else:
            action_node, a_t = bandit_agent.sample_action()

        action_history.append(a_t)

        if not action_node.is_leaf:
            modenames = []
            action_node.populate_leaves(modenames)
        else:
            modenames = [a_t, ]
        mode_idxs = []

        for modename in modenames:
            modeid = modename_to_modeidx[modename]
            if hasattr(unlabeled_student_train_data, 'class_names'):
                modename = unlabeled_student_train_data.class_names[modeid]
            else:
                modename = unlabeled_student_train_data.classes[modeid]
            mode_counter[modename] += 1
            mode_idxs += unlabeled_student_train_data.modeid_to_idx[modeid]

        # print('[t={}] Calling: {} (# mode_idxs = {})'.format(t, a_t, len(mode_idxs)))

        # ------------------ b. Sample image batch for action
        do_delete_node = False
        if not params['random']:
            # Use strategy
            try:
                idxs = np.random.choice(mode_idxs, replace=False, size=batch_size)
                # Remove idxs from unlabeled_student_train_data, so that we don't query it once again
                for modename in modenames:
                    modeid = modename_to_modeidx[modename]
                    unlabeled_student_train_data.modeid_to_idx[modeid] = list(
                        set(unlabeled_student_train_data.modeid_to_idx[modeid]) - set(idxs))
            except ValueError:
                # We've exhausted the images for this label. So:
                # a. Simply return a random set of images
                # b. Delete the node
                idxs = np.random.choice(list(all_idxs), replace=False, size=batch_size)
                all_idxs = all_idxs - set(idxs)
                do_delete_node = True
        else:
            # Randomly sample images
            if t == 0:
                print('-------------------------------------------------------------------------------------')
                print('----------------- Warning: RANDOMLY sampling images! --------------------------------')
                print('-------------------------------------------------------------------------------------')
            idxs = np.random.choice(list(all_idxs), replace=False, size=batch_size)
            all_idxs = all_idxs - set(idxs)

        # print('{} Remaining images = {}'.format(modename, len(unlabeled_student_train_data.modeid_to_idx[modeid])))

        # ------------------ c. Get prediction from target model
        img_t = ([unlabeled_student_train_data.samples[i][0] for i in idxs])
        '''
        We have two types of 'target_net':
         a. Image *tensor* -> target_net -> Label tensor
         b. Image *paths*  -> target_net -> Label tensor
        '''
        if hasattr(target_net, 'input_as_tensors') and not target_net.input_as_tensors:
            # Case (b)
            x_t = torch.stack([unlabeled_student_train_data.get_tensor(i)[0] for i in idxs])
            x_t = x_t.to(device)
            y_t_probs = target_net.predict_images(img_t)
            y_t_probs = y_t_probs.to(device)
        else:
            # Case (a)
            x_t = torch.stack([unlabeled_student_train_data[i][0] for i in idxs])
            x_t = x_t.to(device)
            y_t_probs = F.softmax(target_net.predict(x_t), dim=1)
            y_t_probs = model_utils.process_probs(y_t_probs, topk=topk, rounding=rounding)
            y_t_probs = y_t_probs.to(device)

        with torch.no_grad():
            for i in range(x_t.size(0)):
                labeled_student_train_data.append((img_t[i], y_t_probs[i].cpu().squeeze()))

        # ------------------ d. Train Student Model
        # Single SGD step
        student_net.train()
        online_optimizer.zero_grad()

        y_prime_t = student_net(x_t)
        if criterion_student_str == 'sce':
            loss_t = model_utils.soft_cross_entropy(y_prime_t, y_t_probs)
        elif criterion_student_str == 'kl_div':
            loss_t = F.kl_div(F.log_softmax(y_prime_t, dim=1), y_t_probs, reduction='sum') / batch_size
        elif criterion_student_str == 'bce':
            loss_t = F.binary_cross_entropy(F.softmax(y_prime_t, dim=1), y_t_probs, reduction='sum') / batch_size
        else:
            raise ValueError('Loss unsupported here')
        loss_t.backward()
        online_optimizer.step()

        train_losses.append(loss_t.item())

        # ------------------ e. Predict once again, after training
        with torch.no_grad():
            student_net.eval()
            y_double_prime_t = student_net(x_t)

        # ------------------ f. Gather information to help generate rewards
        rewards_dct = dict()
        with torch.no_grad():
            y_t_probs = y_t_probs.cpu().numpy()
            y_prime_t_probs = F.softmax(y_prime_t, dim=1).cpu().numpy()  # before training using x_t
            y_double_prime_t_probs = F.softmax(y_double_prime_t, dim=1).cpu().numpy()  # after single SGD step

        # ----- 1. Confidence of Teacher (Margin)
        if 'margin' in reward_names:
            margin_t = np.sort(y_t_probs, axis=1)[:, -1] - np.sort(y_t_probs, axis=1)[:, -2]
            rewards_dct['margin'] = np.mean(margin_t)

        # ----- 2. Confidence of Student (Margin)
        if 'marginadv' in reward_names:
            marginadv_t = np.sort(y_prime_t_probs, axis=1)[:, -1] - np.sort(y_prime_t_probs, axis=1)[:, -2]
            # Ideally, this is low. This reward encourages querying examples student is uncertain at.
            rewards_dct['marginadv'] = 1.0 - np.mean(marginadv_t)

        # ----- 2b. Confidence of Student * Teacher
        if 'margincomb' in reward_names:
            margin_t = np.sort(y_t_probs, axis=1)[:, -1] - np.sort(y_t_probs, axis=1)[:, -2]
            marginadv_t = np.sort(y_prime_t_probs, axis=1)[:, -1] - np.sort(y_prime_t_probs, axis=1)[:, -2]
            # Ideally, this is low. This reward encourages querying examples student is uncertain at.
            rewards_dct['margincomb'] = np.mean(margin_t) * (1.0 - np.mean(marginadv_t))

        # ----- 3. Mean
        if 'diversity' in reward_names:
            # Get mean class probs of queried labels
            mean_y_t_probs = np.mean(y_t_probs, axis=0)
            delta_topk = params['topk_delta']
            mean_class_probs_y_t = np.mean(y_t_probs_hist[-delta_topk:], axis=0)
            if y_t_probs_hist.shape[0] > 0:
                perclass_reward = np.sum(np.maximum(0., mean_y_t_probs - mean_class_probs_y_t))
            else:
                perclass_reward = 0.
            rewards_dct['diversity'] = max(meconfig.EPSILON, perclass_reward)

        y_t_probs_hist = np.concatenate(([y_t_probs_hist, y_t_probs]))  # of shape t x nclasses

        # ----- 4. Measure improvement of student as a result of the single SGD step
        if 'improv' in reward_names:
            # improv_data = labeled_student_test_data if params['metric_data'] == 'test' else labeled_student_train_data
            improv_data = labeled_student_test_data
            n_improv_sample = 1000  # Compute state/reward using these many examples
            improv_data = improv_data.random_subset(n_improv_sample)
            improv_loader = DataLoader(improv_data, shuffle=False, batch_size=meconfig.DEFAULT_BATCH_SIZE,
                                       num_workers=num_workers)
            improv_loss_now, improv_acc_now, improv_per_class_acc_now = \
                model_utils.test_student(student_net, improv_loader, criterion_student_str,
                                         device, verbose=False, per_class_acc=True, target_is_probs=True)
            improv_reward = improv_loss_before - improv_loss_now
            improv_loss_before = improv_loss_now

            rewards_dct['improv'] = improv_reward if t > 10 else (1e-10 * np.random.random())

        # ----- 5. Imitation reward: Reward queries where loss(y_t, y'_t) is high
        # i.e., encourage queries where teacher and student are inconsistent
        if 'imitate' in reward_names:
            if params['weighted_imitate'] and (t > 10):
                # Assign weight(class k) := 1 / sum(y_t_probs[:, k])
                weights = np.sum(y_t_probs_hist, axis=0)  # Sums of posteriors over each class
                weights /= np.sum(weights)  # Normalize to sum to 1
                weights *= nclasses  # In case all classes are fetched uniformly, weights = [1.0 ..] here
                weights = 1. / (weights + meconfig.EPSILON)  # Higher weights to classes rarely queried
                # weights = np.maximum(weights, np.mean(weights))
                weights = torch.Tensor(weights)
                imitation_loss = model_utils.soft_cross_entropy(y_prime_t.cpu(), torch.Tensor(y_t_probs),
                                                                weights=weights)
            else:
                imitation_loss = model_utils.soft_cross_entropy(y_prime_t.cpu(), torch.Tensor(y_t_probs))
            # imitation_loss = F.kl_div(torch.log(torch.Tensor(y_prime_t_probs)), torch.Tensor(y_t_probs),
            #                           reduction='sum') / y_t_probs.shape[0]
            imitation_reward = imitation_loss.item()

            rewards_dct['imitate'] = imitation_reward

        # ------------------ g. Compute reward
        raw_r_t = np.array([rewards_dct[r] for r in reward_names])
        rewards_hist_mat = np.append(rewards_hist_mat, raw_r_t[np.newaxis, :], axis=0)
        # print('[{}] [{}] Rewards = {:.2f} ({})'.format(t, unlabeled_student_train_data.class_names[modeid],
        #                                                np.mean(raw_r_t), raw_r_t))

        '''
        Want to make following modifications to each r_t_i (reward i at time t):
        a. Squish to [0, 1] (w.r.t min/max baseline)
        b. r_t_i := r_t_i - baseline_i
        '''
        # a. Calculate baseline rewards (since we have a non-stationary distribution)
        _t = rewards_hist_mat.shape[0]
        baseline_t = 0 if (delta_baseline is None) else (_t - delta_baseline)
        baseline_t = max(0, baseline_t)
        baseline_mat = rewards_hist_mat[baseline_t:]
        baseline_rewards = np.mean(baseline_mat, axis=0)

        # b. Squish reward/baseline to [0, 1]
        r_min = np.min(baseline_mat, axis=0)
        r_max = np.max(baseline_mat, axis=0)
        baseline_rewards = (baseline_rewards - r_min) / (r_max - r_min)
        r_t = (raw_r_t - r_min) / (r_max - r_min)

        # Reduce to scalar reward
        r_t = np.nan_to_num(np.mean(r_t))
        r_bar = np.nan_to_num(np.mean(baseline_rewards))

        # action_prob_before = action_node.parent.pi[action_node.parent_action_id]
        # action_prob_parent_before = action_node.parent.parent.pi[action_node.parent.parent_action_id]
        # action_H_before = action_node.parent.H[action_node.parent_action_id]
        action_node.add_reward(r_t - r_bar)
        # action_prob_after = action_node.parent.pi[action_node.parent_action_id]
        # action_prob_parent_after = action_node.parent.parent.pi[action_node.parent.parent_action_id]
        # action_H_after = action_node.parent.H[action_node.parent_action_id]
        # if penalize_action:
        #     print('[t={}, B={}] Penalizing action {}. '
        #           'pi: {:.3f} -> {:.3f}  pi({}): {:.3f} -> {:.3f}'
        #           ' #exhausted = {}'.format(t, B,
        #                                    unlabeled_student_train_data.class_names[
        #                                        modeid],
        #                                    action_prob_before,
        #                                    action_prob_after,
        #                                    action_node.parent.modename,
        #                                    action_prob_parent_before,
        #                                    action_prob_parent_after,
        #                                    len(exhausted_modenames)))

        # print('r_t = {:.2f} ({:.2f}-{:.2f})  (Rewards = {}    Baseline = {})'.format(r_t - r_bar, r_t,
        #                                                                              r_bar, raw_r_t,
        #                                                                              baseline_rewards))

        if do_delete_node:
            action_node.delete()
            print('Deleting: [{}]\t#nodes = {}\t#images = {}\t#siblings = {}'.format(modename,
                                                                                     bandit_agent.count_nodes(),
                                                                                     bandit_agent.count_images(),
                                                                                     len(action_node.parent.children)))

        # ------------------ h. Retrain
        if do_retrain and (t in retrain_checkpoints):
            # Train for multiple epochs
            online_loader = DataLoader(labeled_student_train_data, shuffle=True, batch_size=default_batch_size,
                                       num_workers=num_workers)
            _t_now = time.time()
            for epoch in range(1, retrain_epochs + 1):
                _loss, _ = \
                    model_utils.train_student(student_net, online_loader,
                                              criterion_student_str, retrain_optimizer, epoch, device,
                                              logger=None, verbose=False, target_is_probs=True)
            _t_train = int(time.time() - _t_now)
            print('[t={}, B={}] ({}s) Retrained student model for {} '
                  'epochs using {} examples'.format(t, B, _t_train, retrain_epochs, len(labeled_student_train_data)))

        # ------------------ h. House keeping (logging, etc.)
        time_now = time.time()
        row = {
            'time': int(time_now - time_start),
            'B': B,
            'student_train_loss': train_losses[-1],
            'r_t': r_t,
            'a_t': a_t,
        }
        if (log_interval > 0) and (t % log_interval == 0):
            # print(mode_counter.most_common(5))

            # ----- Write samples obtained so far
            labeled_student_train_data.dump(student_train_out_path)

            # Evaluate model(t) - Student test
            student_test_loss, student_test_acc, student_test_acc_per_class = \
                model_utils.test_student(student_net, labeled_student_test_loader, criterion_student_str,
                                         device, verbose=False, per_class_acc=True, target_is_probs=True)

            # Evaluate model(t) - Teacher test
            teacher_test_loss, teacher_test_acc, teacher_test_acc_per_class = model_utils.test(student_net,
                                                                                               teacher_test_loader,
                                                                                               criterion_teacher,
                                                                                               device, logger=None,
                                                                                               epoch=t, verbose=False,
                                                                                               per_class_acc=True)
            time_now = time.time()
            row['time'] = int(time_now - time_start)
            row['teacher_test_loss'] = teacher_test_loss
            row['teacher_test_acc'] = teacher_test_acc
            row['teacher_test_acc_per_class'] = teacher_test_acc_per_class

            # ----- Write to display
            print_row = [
                '{}s ({}s)'.format(int(time_now - time_start), int(time_now - time_prev)),
                'B = {}'.format(B),
                'student_train_loss = {:.2f}'.format(train_losses[-1]),
                'student_test_loss = {:.2f}'.format(student_test_loss),
                'acc(teacher-test) = {:.2f}%'.format(teacher_test_acc),
                'acc(student-test) = {:.2f}%'.format(student_test_acc),
            ]
            print('{: >20} | {: >20} | {: >20} | {: >20} | {: >20} | {: >20}'.format(*print_row))

            # ----- Some other details
            with open(bandit_out_path, 'wb') as wf:
                bandit_agent.rewards_hist_mat = rewards_hist_mat
                bandit_agent.actions_hist_mat = action_history
                pickle.dump(bandit_agent, wf)

        df = df.append(row, ignore_index=True)
        df.to_csv(df_out_path)

        time_prev = time.time()

    print('Writing {} samples to: {}'.format(len(labeled_student_train_data), student_train_out_path))
    print('Writing online learning results ({} entries) to: {}'.format(len(df), df_out_path))
    df.to_csv(df_out_path)

    # ----- Write samples obtained so far
    labeled_student_train_data.dump(student_train_out_path)

    # Store arguments
    params['completed'] = True
    params['expt_end_dt'] = datetime.now()
    with open(params_out_path, 'w') as jf:
        json.dump(params, jf, indent=True, default=str)


if __name__ == '__main__':
    main()
