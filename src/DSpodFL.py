from random import uniform, betavariate, random

import networkx as nx
from scipy.stats import truncnorm

import utils
from Agent import Agent


class DSpodFL:
    def __init__(self,
                 model_name: str,
                 dataset_name: str,
                 num_epochs: int,
                 num_agents: int,
                 graph_connectivity: float,
                 labels_per_agent: int,
                 batch_size: int,
                 learning_rate: float,
                 prob_aggr_type: str,
                 prob_sgd_type: str,
                 sim_type: str,
                 prob_sgd_dist_params,
                 prob_aggr_dist_params,
                 termination_delay: float,
                 DandB,
                 cta: bool,
                 comm_weight: float,
                 is_async: bool):
        self.model_name = model_name
        self.num_epochs = num_epochs
        self.num_agents = num_agents
        self.graph_connectivity = graph_connectivity
        self.labels_per_agent = labels_per_agent
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.prob_aggr_type = prob_aggr_type
        self.prob_sgd_type = prob_sgd_type
        self.sim_type = sim_type
        self.prob_sgd_dist_params = prob_sgd_dist_params
        self.prob_aggr_dist_params = prob_aggr_dist_params
        self.termination_delay = termination_delay
        self.cta = cta
        self.comm_weight = comm_weight
        self.is_async = is_async

        self.num_classes, transform, self.num_channels = utils.aux_info(dataset_name, model_name)
        self.train_set, self.test_set, self.input_dim = utils.dataset_info(dataset_name, transform)
        print(f"train={len(self.train_set)}, test={len(self.test_set)}")

        self.prob_sgds_bank = self.generate_prob_sgds_bank()
        self.prob_aggrs_bank = self.generate_prob_aggrs_bank()

        self.graph = self.generate_graph()
        self.initial_prob_sgds = self.generate_prob_sgds(is_initial=True)
        self.initial_prob_aggrs = self.generate_prob_aggrs(is_initial=True)
        self.prob_sgds = self.generate_prob_sgds()
        self.prob_aggrs = self.generate_prob_aggrs()
        train_sets = utils.generate_train_sets(self.train_set, self.num_agents, self.num_classes, self.labels_per_agent)
        models, criterion, self.model_dim = self.generate_models()

        self.agents = self.generate_agents(self.initial_prob_sgds, models, criterion, train_sets)
        self.DandB = utils.determine_DandB(DandB, self.initial_prob_sgds, self.initial_prob_aggrs)
        print(self.DandB)

    def generate_graph(self):
        while True:
            graph = nx.random_geometric_graph(self.num_agents, self.graph_connectivity)
            if nx.is_k_edge_connected(graph, 1):
                break
        return graph

    def generate_prob_sgds_bank(self):
        dynamic_count = len(self.train_set) // self.num_agents // 1000

        if self.prob_sgd_type == 'random_dynamic':
            return [
                [uniform(self.prob_sgd_dist_params[0], self.prob_sgd_dist_params[1]) for _ in range(self.num_agents)]
                for _
                in range(dynamic_count)]
        elif self.prob_sgd_type == 'beta_dynamic':
            return [[betavariate(self.prob_sgd_dist_params[0], self.prob_sgd_dist_params[1]) for _ in
                     range(self.num_agents)]
                    for _ in range(dynamic_count)]
        else:
            return None

    def generate_prob_aggrs_bank(self):
        dynamic_count = len(self.train_set) // self.num_agents // 1000

        if self.prob_aggr_type == 'random_dynamic':
            return [
                [[uniform(self.prob_aggr_dist_params[0], self.prob_aggr_dist_params[1]) for _ in range(self.num_agents)]
                 for _ in
                 range(self.num_agents)] for _ in range(dynamic_count)]
        elif self.prob_aggr_type == 'beta_dynamic':
            return [
                [[betavariate(self.prob_aggr_dist_params[0], self.prob_aggr_dist_params[1]) for _ in
                  range(self.num_agents)] for _
                 in range(self.num_agents)] for _ in range(dynamic_count)]
        else:
            return None

    def generate_prob_sgds(self, is_initial=False, dynamic_index=0):
        if is_initial:
            if self.prob_sgd_type == 'random':
                return [uniform(self.prob_sgd_dist_params[0], self.prob_sgd_dist_params[1]) for _ in
                        range(self.num_agents)]
            elif self.prob_sgd_type == 'beta':
                return [betavariate(self.prob_sgd_dist_params[0], self.prob_sgd_dist_params[1]) for _ in
                        range(self.num_agents)]
            elif self.prob_sgd_type == 'truncnorm':
                return [truncnorm.rvs(-self.prob_sgd_dist_params[0] / self.prob_sgd_dist_params[1],
                                      (1 - self.prob_sgd_dist_params[0]) / self.prob_sgd_dist_params[1],
                                      loc=self.prob_sgd_dist_params[0], scale=self.prob_sgd_dist_params[1]) for _ in
                        range(self.num_agents)]
            elif self.prob_sgd_type == 'bimodal':
                return [truncnorm.rvs(-self.prob_sgd_dist_params[0][0] / self.prob_sgd_dist_params[0][1],
                                      (1 - self.prob_sgd_dist_params[0][0]) / self.prob_sgd_dist_params[0][1],
                                      loc=self.prob_sgd_dist_params[0][0],
                                      scale=self.prob_sgd_dist_params[0][1]) if random() <= 0.5 else truncnorm.rvs(
                    -self.prob_sgd_dist_params[1][0] / self.prob_sgd_dist_params[1][1],
                    (1 - self.prob_sgd_dist_params[1][0]) / self.prob_sgd_dist_params[1][1],
                    loc=self.prob_sgd_dist_params[1][0], scale=self.prob_sgd_dist_params[1][1]) for _ in
                        range(self.num_agents)]
            elif self.prob_sgd_type in ['random_dynamic', 'beta_dynamic', 'full', 'zero']:
                return self.prob_sgds_bank[dynamic_index]
        elif self.prob_sgd_type in ['random', 'beta', 'truncnorm', 'bimodal', 'random_dynamic', 'beta_dynamic']:
            return self.initial_prob_sgds
        elif self.prob_sgd_type == 'full':
            return [1 for _ in range(self.num_agents)]
        elif self.prob_sgd_type == 'zero':
            return [0 for _ in range(self.num_agents)]

    def generate_prob_aggrs(self, is_initial=False, dynamic_index=0):
        prob_aggrs = [[None for _ in range(self.num_agents)] for _ in range(self.num_agents)]
        for i in range(self.num_agents):
            for j in range(self.num_agents):
                # if j in self.graph.adj[i] and prob_aggrs[i][j] is None:
                if prob_aggrs[i][j] is None:
                    if is_initial:
                        if self.prob_aggr_type == 'random':
                            prob_aggrs[i][j] = prob_aggrs[j][i] = uniform(self.prob_aggr_dist_params[0],
                                                                          self.prob_aggr_dist_params[1])
                        elif self.prob_aggr_type == 'beta':
                            prob_aggrs[i][j] = prob_aggrs[j][i] = betavariate(self.prob_aggr_dist_params[0],
                                                                              self.prob_aggr_dist_params[1])
                        elif self.prob_aggr_type == 'truncnorm':
                            prob_aggrs[i][j] = prob_aggrs[j][i] = truncnorm.rvs(
                                -self.prob_aggr_dist_params[0] / self.prob_aggr_dist_params[1],
                                (1 - self.prob_aggr_dist_params[0]) / self.prob_aggr_dist_params[1],
                                loc=self.prob_aggr_dist_params[0], scale=self.prob_aggr_dist_params[1])
                        elif self.prob_aggr_type == 'bimodal':
                            prob_aggrs[i][j] = prob_aggrs[j][i] = truncnorm.rvs(
                                -self.prob_sgd_dist_params[0][0] / self.prob_sgd_dist_params[0][1],
                                (1 - self.prob_sgd_dist_params[0][0]) / self.prob_sgd_dist_params[0][1],
                                loc=self.prob_sgd_dist_params[0][0],
                                scale=self.prob_sgd_dist_params[0][1]) if random() <= 0.5 else truncnorm.rvs(
                                -self.prob_sgd_dist_params[1][0] / self.prob_sgd_dist_params[1][1],
                                (1 - self.prob_sgd_dist_params[1][0]) / self.prob_sgd_dist_params[1][1],
                                loc=self.prob_sgd_dist_params[1][0], scale=self.prob_sgd_dist_params[1][1])
                        elif self.prob_aggr_type in ['random_dynamic', 'beta_dynamic', 'full', 'zero']:
                            prob_aggrs[i][j] = prob_aggrs[j][i] = self.prob_aggrs_bank[dynamic_index][i][j]
                    elif self.prob_aggr_type in ['random', 'beta', 'truncnorm', 'bimodal', 'random_dynamic', 'beta_dynamic']:
                        prob_aggrs[i][j] = prob_aggrs[j][i] = self.initial_prob_aggrs[i][j]
                    elif self.prob_aggr_type == 'full':
                        prob_aggrs[i][j] = prob_aggrs[j][i] = 1
                    elif self.prob_aggr_type == 'zero':
                        prob_aggrs[i][j] = prob_aggrs[j][i] = 0
        return [[p / self.comm_weight for p in p_row] for p_row in prob_aggrs]

    def generate_models(self):
        models, criterion, model_dim = [], None, None
        for _ in range(self.num_agents):
            model, criterion, model_dim = utils.model_info(self.model_name, self.input_dim,
                                                           self.num_classes, self.num_channels)
            models.append(model)
        models = [models[0] for _ in models]
        return models, criterion, model_dim

    def generate_agents(self, prob_sgds, models, criterion, train_sets):
        agents = []
        for i in range(self.num_agents):
            agent_i = Agent(
                initial_model=models[i],
                criterion=criterion,
                train_set=train_sets[i],
                test_set=self.test_set,
                batch_size=self.batch_size,
                learning_rate=self.learning_rate,
                prob_sgd=prob_sgds[i],
                cta=self.cta,
                comm_weight=self.comm_weight,
                is_async=self.is_async
            )
            agents.append(agent_i)

        for i in range(self.num_agents):
            for j in list(self.graph.adj[i]):
                agents[i].add_neighbor(agents[j], self.prob_aggrs[i][j], self.initial_prob_aggrs[i][j])
        return agents

    def run(self):
        # num_iters = 2  # comment this line (this was used for testing)
        num_iters = len(self.train_set) // self.num_agents
        total_iter = 0

        iters, iters_sampled = [], []
        losses, accuracies = [], []
        cpu_useds, cpu_useds_cumsum, cpu_utilizations = [], [], []
        processing_time_useds, processing_time_useds_cumsum, processing_time_utilizations = [], [], []
        bandwidth_useds, bandwidth_useds_cumsum, bandwidth_utilizations = [], [], []
        transmission_time_useds, transmission_time_useds_cumsum, transmission_time_utilizations = [], [], []
        delay_useds, delay_useds_cumsum, delay_utilizations = [], [], []
        pure_runtimes, pure_runtimes_cumsum, runtimes, runtimes_cumsum = [], [], [], []

        for k in range(self.num_epochs):
            print(f"epoch: {k}")

            for i in range(num_iters):
                total_iter = k * num_iters + i
                # print(f"epoch: {k}, iter: {i}, total_iter={total_iter}")

                if self.sim_type == 'dynamic_probs' and i % 1000 == 0:
                    self.reset_prob_sgd_dist_params(prob_sgd_dist_params=None, dynamic_index=i // 1000)
                    self.reset_prob_aggr_dist_params(prob_aggr_dist_params=None, dynamic_index=i // 1000)

                loss = 0
                cpu_used, max_cpu_usable = 0, 0
                bandwidth_used, max_bandwidth_usable = 0, 0
                transmission_time_used, max_transmission_time_usable = 0, 0
                processing_time_used, max_processing_time_usable = 0, 0
                delay_used, max_delay_usable = 0, 0
                runtime, pure_runtime = 0, 0

                for j in range(self.num_agents):
                    self.agents[j].run_step1()

                for j in range(self.num_agents):
                    loss += float(self.agents[j].get_loss())
                    cpu_used += self.agents[j].cpu_used()
                    max_cpu_usable += self.agents[j].max_cpu_usable()
                    processing_time_used += self.agents[j].processing_time_used()
                    max_processing_time_usable += self.agents[j].max_processing_time_usable()
                    bandwidth_used += self.agents[j].bandwidth_used()
                    max_bandwidth_usable += self.agents[j].max_bandwidth_usable()
                    transmission_time_used += self.agents[j].transmission_time_used()
                    max_transmission_time_usable += self.agents[j].max_transmission_time_usable()

                    delay_used += self.agents[j].delay_used()
                    max_delay_usable += self.agents[j].max_delay_usable()
                    pure_runtime += self.agents[j].get_pure_runtime()
                    runtime += self.agents[j].runtime_used()

                for j in range(self.num_agents):
                    self.agents[j].run_step2()

                iters.append(total_iter)
                losses.append(loss / self.num_agents)
                cpu_utilizations.append(cpu_used / max_cpu_usable)
                processing_time_utilizations.append(processing_time_used / max_processing_time_usable)
                bandwidth_utilizations.append(bandwidth_used / max_bandwidth_usable)
                transmission_time_utilizations.append(transmission_time_used / max_transmission_time_usable)
                delay_utilizations.append(delay_used / max_delay_usable)

                cpu_useds.append(cpu_used / self.num_agents)
                processing_time_useds.append(processing_time_used / self.num_agents)
                bandwidth_useds.append(bandwidth_used / self.num_agents)
                transmission_time_useds.append(transmission_time_used / self.num_agents)
                delay_useds.append(delay_used / self.num_agents)

                pure_runtimes.append(pure_runtime / self.num_agents)
                runtimes.append(runtime / self.num_agents)
                if total_iter == 0:
                    cpu_useds_cumsum.append(cpu_useds[-1])
                    processing_time_useds_cumsum.append(processing_time_useds[-1])
                    bandwidth_useds_cumsum.append(bandwidth_useds[-1])
                    transmission_time_useds_cumsum.append(transmission_time_useds[-1])
                    delay_useds_cumsum.append(delay_useds[-1])
                    pure_runtimes_cumsum.append(pure_runtimes[-1])
                    runtimes_cumsum.append(runtimes[-1])
                else:
                    cpu_useds_cumsum.append(cpu_useds_cumsum[-1] + cpu_useds[-1])
                    processing_time_useds_cumsum.append(
                        processing_time_useds_cumsum[-1] + processing_time_useds[-1])
                    bandwidth_useds_cumsum.append(bandwidth_useds_cumsum[-1] + bandwidth_useds[-1])
                    transmission_time_useds_cumsum.append(
                        transmission_time_useds_cumsum[-1] + transmission_time_useds[-1])
                    delay_useds_cumsum.append(delay_useds_cumsum[-1] + delay_useds[-1])
                    pure_runtimes_cumsum.append(pure_runtimes_cumsum[-1] + pure_runtimes[-1])
                    runtimes_cumsum.append(runtimes_cumsum[-1] + runtimes[-1])

                if self.accuracy_calculation_condition(total_iter, delay_useds_cumsum[-1]):
                    accuracy = 0
                    bw_util = 0
                    cpu_util = 0

                    for j in range(self.num_agents):
                        self.agents[j].calculate_accuracy()

                    for j in range(self.num_agents):
                        accuracy += self.agents[j].get_accuracy()
                        curr_bw = self.agents[j].get_aggregation_count() / (
                                self.agents[j].get_degree() * (k * num_iters + i + 1))
                        curr_cpu = self.agents[j].get_data_processed() / (self.batch_size * (k * num_iters + i + 1))
                        bw_util += curr_bw
                        cpu_util += curr_cpu
                        # print(f"Agent {j}: accuracy = {self.agents[j].get_accuracy()}, bw_util = {curr_bw}, "
                        #       f"cpu_util = {curr_cpu}")

                    accuracies.append(accuracy / self.num_agents)
                    # print(f"iter = {i}, avg accuracy = {accuracies[-1]}, avg bw_util = {bw_util / self.num_agents}, "
                    #       f"avg cpu_util = {cpu_util / self.num_agents}")
                    iters_sampled.append(total_iter)

                if self.change_probs_condition(total_iter):
                    if self.prob_sgd_type == 'full':
                        self.reset_prob_sgds('zero')
                        self.reset_prob_aggrs('full')
                    else:
                        self.reset_prob_sgds('full')
                        self.reset_prob_aggrs('zero')

                if self.termination_condition(total_iter, delay_useds_cumsum[-1]):
                    break
            if self.termination_condition(total_iter, delay_useds_cumsum[-1]):
                break

        log1 = {"iters": iters,
                "losses": losses,
                "cpu_useds": cpu_useds,
                "cpu_useds_cumsum": cpu_useds_cumsum,
                "cpu_utilizations": cpu_utilizations,
                "processing_time_useds": processing_time_useds,
                "processing_time_useds_cumsum": processing_time_useds_cumsum,
                "processing_time_utilizations": processing_time_utilizations,
                "bandwidth_useds": bandwidth_useds,
                "bandwidth_useds_cumsum": bandwidth_useds_cumsum,
                "bandwidth_utilizations": bandwidth_utilizations,
                "transmission_time_useds": transmission_time_useds,
                "transmission_time_useds_cumsum": transmission_time_useds_cumsum,
                "transmission_time_utilizations": transmission_time_utilizations,
                "delay_useds": delay_useds,
                "delay_useds_cumsum": delay_useds_cumsum,
                "delay_utilizations": delay_utilizations,
                "pure_runtimes": pure_runtimes,
                "pure_runtimes_cumsum": pure_runtimes_cumsum,
                "runtimes": runtimes,
                "runtimes_cumsum": runtimes_cumsum}

        log2 = {"iters_sampled": iters_sampled,
                "accuracies": accuracies}
        return [log1, log2]

    def accuracy_calculation_condition(self, total_iter, total_delay):
        cond_eff = self.sim_type in ['eff', 'dynamic_probs'] and total_iter % (
                10 ** (len(str(total_iter)) - 1) / 2) == 0
        cond_graph_conn = self.sim_type in ['graph_conn', 'data_dist', 'alpha_beta', 'alpha_beta_v2', 'alpha_beta_aggr',
                                            'num_agents', 'truncnorm', 'bimodal', 'cta', 'learning_rate'] and total_delay >= self.termination_delay
        return cond_eff or cond_graph_conn

    def termination_condition(self, total_iter, total_delay):
        cond = self.accuracy_calculation_condition(total_iter, total_delay)
        return cond and self.sim_type in ['graph_conn', 'data_dist', 'alpha_beta', 'alpha_beta_v2', 'alpha_beta_aggr',
                                          'num_agents', 'truncnorm', 'bimodal', 'cta', 'learning_rate']

    def change_probs_condition(self, total_iter):
        D, B = self.DandB
        cond = self.prob_sgd_type == 'full' and self.prob_aggr_type == 'zero'
        cond = cond or (self.prob_sgd_type == 'zero' and self.prob_aggr_type == 'full')
        return cond and ((total_iter + 1) % (D + B) == 0 or (total_iter + 1) % (D + B) == D)

    def reset(self, graph_connectivity=0.4, labels_per_agent=None, prob_aggr_type='random', prob_sgd_type='random',
              sim_type='eff', prob_sgd_dist_params=(0, 1), prob_aggr_dist_params=(0, 1), num_agents=10, learning_rate=0.01, is_async=False):
        if graph_connectivity != self.graph_connectivity:
            self.reset_graph(graph_connectivity)
        if labels_per_agent != self.labels_per_agent:
            self.reset_train_sets(labels_per_agent)
        if prob_aggr_type != self.prob_aggr_type:
            self.reset_prob_aggrs(prob_aggr_type)
        if prob_sgd_type != self.prob_sgd_type:
            self.reset_prob_sgds(prob_sgd_type)
        if sim_type != self.sim_type:
            self.reset_sim_type(sim_type)
        if prob_sgd_dist_params != self.prob_sgd_dist_params:
            self.reset_prob_sgd_dist_params(prob_sgd_dist_params)
        if prob_aggr_dist_params != self.prob_aggr_dist_params:
            self.reset_prob_aggr_dist_params(prob_aggr_dist_params)
        if num_agents != self.num_agents:
            self.reset_num_agents(num_agents)
        if learning_rate != self.learning_rate:
            self.reset_learning_rate(learning_rate)
        if is_async != is_async:
            self.reset_async(is_async)

        for agent in self.agents:
            # model, _, _ = utils.model_info(self.model_name, self.input_dim, self.num_classes, self.num_channels)
            # agent.reset(model=model)
            agent.reset()

    def reset_graph(self, graph_connectivity):
        self.graph_connectivity = graph_connectivity
        self.graph = self.generate_graph()

        for i in range(self.num_agents):
            self.agents[i].clear_neighbors()
            for j in list(self.graph.adj[i]):
                self.agents[i].add_neighbor(self.agents[j], self.prob_aggrs[i][j], self.initial_prob_aggrs[i][j])

    def reset_train_sets(self, labels_per_agent):
        self.labels_per_agent = labels_per_agent
        train_sets = utils.generate_train_sets(self.train_set, self.num_agents, self.num_classes, self.labels_per_agent)
        for i in range(self.num_agents):
            self.agents[i].set_train_set(train_sets[i])

    def reset_prob_aggrs(self, prob_aggr_type):
        self.prob_aggr_type = prob_aggr_type
        self.prob_aggrs = self.generate_prob_aggrs()

        for i in range(self.num_agents):
            for j in list(self.graph.adj[i]):
                self.agents[i].set_prob_aggr(self.agents[j], self.prob_aggrs[i][j])

    def reset_prob_sgds(self, prob_sgd_type):
        self.prob_sgd_type = prob_sgd_type
        prob_sgds = self.generate_prob_sgds()

        for i in range(self.num_agents):
            self.agents[i].set_prob_sgd(prob_sgds[i])

    def reset_sim_type(self, sim_type):
        self.sim_type = sim_type

    def reset_prob_sgd_dist_params(self, prob_sgd_dist_params=None, dynamic_index=0):
        if prob_sgd_dist_params:
            self.prob_sgd_dist_params = prob_sgd_dist_params
            self.initial_prob_sgds = self.generate_prob_sgds(is_initial=True)
            self.prob_sgds = self.generate_prob_sgds()
        else:
            self.initial_prob_sgds = self.generate_prob_sgds(is_initial=True, dynamic_index=dynamic_index)
            self.prob_sgds = self.generate_prob_sgds(dynamic_index=dynamic_index)

        for i in range(self.num_agents):
            self.agents[i].set_prob_sgd(self.prob_sgds[i])
            self.agents[i].set_initial_prob_sgd(self.initial_prob_sgds[i])

    def reset_prob_aggr_dist_params(self, prob_aggr_dist_params=None, dynamic_index=0):
        if prob_aggr_dist_params:
            self.prob_aggr_dist_params = prob_aggr_dist_params
            self.initial_prob_aggrs = self.generate_prob_aggrs(is_initial=True)
            self.prob_aggrs = self.generate_prob_aggrs()
        else:
            self.initial_prob_aggrs = self.generate_prob_aggrs(is_initial=True, dynamic_index=dynamic_index)
            self.prob_aggrs = self.generate_prob_aggrs(dynamic_index=dynamic_index)

        for i in range(self.num_agents):
            for j in list(self.graph.adj[i]):
                self.agents[i].set_prob_aggr(self.agents[j], self.prob_aggrs[i][j])
                self.agents[i].set_initial_prob_aggr(self.agents[j], self.initial_prob_aggrs[i][j])

    def reset_num_agents(self, num_agents):
        for agent in self.agents:
            agent.release_model()

        self.num_agents = num_agents
        self.graph = self.generate_graph()
        self.initial_prob_sgds = self.generate_prob_sgds(is_initial=True)
        self.prob_aggrs = self.initial_prob_aggrs = self.generate_prob_aggrs(is_initial=True)
        train_sets = utils.generate_train_sets(self.train_set, self.num_agents, self.num_classes, self.labels_per_agent)

        models, criterion, self.model_dim = self.generate_models()
        self.agents = self.generate_agents(self.initial_prob_sgds, models, criterion, train_sets)

    def reset_learning_rate(self, learning_rate):
        for agent in self.agents:
            agent.set_learning_rate(learning_rate)
        self.learning_rate = learning_rate

    def reset_async(self, is_async):
        for agent in self.agents:
            agent.set_async(is_async)
        self.is_async = is_async
