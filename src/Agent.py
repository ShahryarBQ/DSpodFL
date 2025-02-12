import time
from copy import deepcopy
from random import choices, random

import torch
from torch import optim
from torch.utils.data import DataLoader

import utils


class Agent:
    def __init__(self,
                 initial_model,
                 criterion,
                 train_set,
                 test_set,
                 batch_size: int,
                 learning_rate: float,
                 prob_sgd: float,
                 cta: bool,
                 comm_weight: float,
                 is_async: bool):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.initial_model = initial_model
        self.criterion = criterion
        self.train_set = train_set
        self.test_set = test_set
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.prob_sgd = self.initial_prob_sgd = prob_sgd
        self.cta = cta
        self.comm_weight = comm_weight
        self.is_async = is_async

        self.w = deepcopy(self.initial_model).to(self.device)
        self.len_params = len(list(initial_model.parameters()))

        self.neighbors = []
        self.aggregation_neighbors = []
        self.gradient = None
        self.aggregation = None
        self.loss = None
        self.accuracy = 0
        self.data_processed = None
        self.aggregation_count = None
        self.runtime = None
        self.pure_runtime = None
        self.v = 0
        self.old_gradient = [0 for _ in range(self.len_params)]

    def run_step1(self):
        self.pure_runtime = 0
        start = time.time()

        self.aggregation = [0 for _ in range(self.len_params)]
        self.aggregation_neighbors = []
        for neighbor in self.neighbors:
            if neighbor['v_hat'] == 1:
                self.aggregation_neighbors.append(neighbor)
        if len(self.aggregation_neighbors) != 0:
            self.aggregation = self.event_aggregation()

        end = time.time()
        self.pure_runtime = end - start

    def run_step2(self):
        start = time.time()

        self.w = self.w.to(self.device)
        self.gradient = [0 for _ in range(self.len_params)]

        # added later for cta
        if not self.cta:
            if self.v == 1:
                self.gradient = self.event_data(choices(self.train_set, k=self.batch_size))
                if self.is_async:
                    temp = deepcopy(self.old_gradient)
                    self.old_gradient = deepcopy(self.gradient)
                    self.gradient = temp

            with torch.no_grad():
                param_idx = 0
                for param in self.w.parameters():
                    param.data += self.aggregation[param_idx] - self.gradient[param_idx]
                    param_idx += 1
        else:
            with torch.no_grad():
                param_idx = 0
                for param in self.w.parameters():
                    param.data += self.aggregation[param_idx]
                    param_idx += 1

            if self.v == 1:
                self.gradient = self.event_data(choices(self.train_set, k=self.batch_size))
            with torch.no_grad():
                param_idx = 0
                for param in self.w.parameters():
                    param.data -= self.gradient[param_idx]
                    param_idx += 1

        self.v = 0
        if random() <= self.prob_sgd:
            self.v = 1

        for neighbor in self.neighbors:
            neighbor['vhat'] = 0
            if random() <= neighbor['prob_aggr']:
                neighbor['v_hat'] = 1
                neighbor['agent'].set_v_hat(self, 1)

        end = time.time()
        self.pure_runtime += end - start

    def event_data(self, data):
        self.data_processed += self.batch_size
        return self.gradient_descent(data)

    def event_aggregation(self):
        aggregation = [0 for _ in range(self.len_params)]
        for neighbor in self.aggregation_neighbors:
            aggregation_weight = self.calculate_aggregation_weight(neighbor['agent'])

            param_idx = 0
            neighbor_w = neighbor['agent'].get_w().to(self.device)
            for param, param_neighbor in zip(self.w.parameters(), neighbor_w.parameters()):
                aggregation[param_idx] += aggregation_weight * (param_neighbor.data - param.data)
                param_idx += 1

        self.aggregation_count += len(self.aggregation_neighbors)
        return aggregation

    def gradient_descent(self, data):
        # We do the update on a temporary model, so that we can do the gradient descent
        # and the aggregation at the same iteration later.
        w2 = deepcopy(self.w).to(self.device)
        w2.train()

        train_loader = DataLoader(
            data,
            batch_size=self.batch_size,
            shuffle=False
        )
        dataX, dataY = next(iter(train_loader))
        dataX, dataY = dataX.to(self.device), dataY.to(self.device)

        optimizer = optim.SGD(w2.parameters(), lr=self.learning_rate)
        optimizer.zero_grad()
        output = w2(dataX)
        loss = self.criterion(output, dataY)
        loss.backward()
        optimizer.step()

        gradient = [None for _ in range(self.len_params)]

        param_idx = 0
        for param, param2 in zip(self.w.parameters(), w2.parameters()):
            gradient[param_idx] = param.data - param2.data
            param_idx += 1

        self.loss = loss
        return gradient

    def calculate_aggregation_weight(self, neighbor_agent):
        return 1 / (1 + max(self.get_degree(), neighbor_agent.get_degree()))

    def calculate_accuracy(self):
        self.accuracy = utils.calculate_accuracy(self.w, self.test_set)

    def reset(self, model=None, prob_sgd=None):
        # Agent-based properties
        if prob_sgd is not None:
            self.prob_sgd = prob_sgd

        # Learning-based parameters
        if model is not None:
            self.w = model.to(self.device)  # generate new random weights
        else:
            self.w = deepcopy(self.initial_model).to(self.device)  # reuse initial model every time
        self.loss = 0

        # Aggregation-based parameters
        self.v = 0
        for neighbor in self.neighbors:
            neighbor['v_hat'] = 0

        # Counters
        self.data_processed = 0
        self.aggregation_count = 0
        self.runtime = 0

    def cpu_used(self):
        return self.v

    @staticmethod
    def max_cpu_usable():
        return 1

    def processing_time_used(self):
        return self.cpu_used() / self.initial_prob_sgd
        # return self.cpu_used() / self.initial_prob_sgd * (self.len_params * 10 * self.batch_size / 2e9)

    def max_processing_time_usable(self):
        return self.max_cpu_usable() / self.initial_prob_sgd
        # return self.max_cpu_usable() / self.initial_prob_sgd * (self.len_params * 10 * self.batch_size / 2e9)

    def bandwidth_used(self):
        return len(self.aggregation_neighbors) * self.len_params

    def max_bandwidth_usable(self):
        return self.get_degree() * self.len_params

    def transmission_time_used(self):
        transmission_time_used = 0
        for neighbor in self.aggregation_neighbors:
            transmission_time_used += 1 / neighbor['initial_prob_aggr']
            # transmission_time_used += 1 / neighbor['initial_prob_aggr'] * (self.len_params / 2e6)
        return transmission_time_used / self.get_degree()
        # return transmission_time_used / len(self.aggregation_neighbors)

    def max_transmission_time_usable(self):
        transmission_time_used = 0
        for neighbor in self.neighbors:
            transmission_time_used += 1 / neighbor['initial_prob_aggr']
            # transmission_time_used += 1 / neighbor['initial_prob_aggr'] * (self.len_params / 2e6)
        return transmission_time_used / self.get_degree()

    def delay_used(self):
        return self.processing_time_used() + self.comm_weight * self.transmission_time_used()

    def max_delay_usable(self):
        return self.max_processing_time_usable() + self.comm_weight * self.max_transmission_time_usable()

    def get_pure_runtime(self):
        return self.pure_runtime

    def runtime_used(self):
        return self.pure_runtime / self.initial_prob_sgd

    def add_neighbor(self, agent, prob_aggr, initial_prob_aggr):
        self.neighbors.append({'agent': agent, 'prob_aggr': prob_aggr,
                               'initial_prob_aggr': initial_prob_aggr, 'v_hat': 0})

    def clear_neighbors(self):
        self.neighbors = []

    def find_neighbor(self, neighbor_agent):
        for neighbor in self.neighbors:
            if neighbor_agent is neighbor['agent']:
                return neighbor
        return None

    def release_model(self):
        torch.cuda.empty_cache()
        self.w = self.w.to('cpu')

    def get_degree(self):
        return len(self.neighbors)

    def get_w(self):
        return self.w

    def get_v_hat(self, neighbor_agent):
        return self.find_neighbor(neighbor_agent)['v_hat']

    def get_loss(self):
        return self.loss

    def get_accuracy(self):
        return self.accuracy

    def get_aggregation_count(self):
        return self.aggregation_count

    def get_aggregation_neighbors_count(self):
        return len(self.aggregation_neighbors)

    def get_data_processed(self):
        return self.data_processed

    def set_v_hat(self, neighbor_agent, v_hat):
        self.find_neighbor(neighbor_agent)['v_hat'] = v_hat

    def set_train_set(self, train_set):
        self.train_set = train_set

    def set_prob_sgd(self, prob_sgd):
        self.prob_sgd = prob_sgd

    def set_initial_prob_sgd(self, initial_prob_sgd):
        self.initial_prob_sgd = initial_prob_sgd

    def set_prob_aggr(self, neighbor_agent, prob_aggr):
        self.find_neighbor(neighbor_agent)['prob_aggr'] = prob_aggr

    def set_initial_prob_aggr(self, neighbor_agent, initial_prob_aggr):
        self.find_neighbor(neighbor_agent)['initial_prob_aggr'] = initial_prob_aggr

    def set_learning_rate(self, lr):
        self.learning_rate = lr

    def set_async(self, is_async):
        self.is_async = is_async
