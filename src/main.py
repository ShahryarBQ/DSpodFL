import os
import random

import utils
from DSpodFL import DSpodFL


class Simulations:
    def __init__(self, seed):
        self.model_name = 'SVM'
        self.dataset_name = 'FMNIST'
        self.num_epochs = 1
        self.num_agents = 10
        self.graph_connectivity = 0.4
        self.labels_per_agent = 10
        batch_size = 16
        self.learning_rate = 0.01
        self.prob_sgd_type = 'beta'
        self.prob_aggr_type = 'beta'
        self.sim_type = 'learning_rate'

        # beta: (alpha, beta), random: (min, max), truncnorm: (mu, sigma), bimodal: ((mu1, sigma1), (mu2, sigma2))
        self.prob_sgd_dist_params = (0.5, 0.5)
        self.prob_aggr_dist_params = (0.5, 0.5)
        termination_delay = 500 if self.dataset_name == 'FMNIST' else 5000
        DandB = (None, 1)
        self.cta = False
        comm_weight = 1
        self.is_async = False
        self.seed = seed

        self.simulation_count = None
        self.simulation_counter = None

        if self.sim_type == 'dynamic_probs':
            self.prob_aggr_type = f"{self.prob_aggr_type}_dynamic"
            self.prob_sgd_type = f"{self.prob_sgd_type}_dynamic"

        self.dSpodFL = DSpodFL(
            self.model_name,
            self.dataset_name,
            self.num_epochs,
            self.num_agents,
            self.graph_connectivity,
            self.labels_per_agent,
            batch_size,
            self.learning_rate,
            self.prob_aggr_type,
            self.prob_sgd_type,
            self.sim_type,
            self.prob_sgd_dist_params,
            self.prob_aggr_dist_params,
            termination_delay,
            DandB,
            self.cta,
            comm_weight,
            self.is_async
        )

    def run(self):
        if self.sim_type == 'eff':
            self.efficiency_experiment()
        elif self.sim_type == 'graph_conn':
            self.graph_connectivity_experiment()
        elif self.sim_type == 'data_dist':
            self.data_distribution_experiment()
        elif self.sim_type == 'alpha_beta':
            self.alpha_beta_experiment()
        elif self.sim_type == 'alpha_beta_v2':
            self.alpha_beta_experiment()
        elif self.sim_type == 'alpha_beta_aggr':
            self.alpha_beta_aggr_experiment()
        elif self.sim_type == 'num_agents':
            self.num_agents_experiment()
        elif self.sim_type == 'dynamic_probs':
            self.dynamic_probs_experiment()
        elif self.sim_type == 'truncnorm':
            self.truncnorm_experiment()
        elif self.sim_type == 'bimodal':
            self.bimodal_experiment()
        elif self.sim_type == 'cta':
            self.cta_experiment()
        elif self.sim_type == 'learning_rate':
            self.learning_rate_experiment()

    def efficiency_experiment(self):
        self.simulation_count = 5
        self.simulation_counter = 1

        prob_aggr_types = [self.prob_aggr_type, 'full']
        prob_sgd_types = [self.prob_sgd_type, 'full']
        for prob_aggr_type in prob_aggr_types:
            for prob_sgd_type in prob_sgd_types:
                self.simulate_and_save_results(self.graph_connectivity, self.labels_per_agent, prob_aggr_type,
                                               prob_sgd_type, self.prob_sgd_dist_params, self.prob_aggr_dist_params,
                                               self.num_agents, self.cta, self.learning_rate, self.is_async)
        self.simulate_and_save_results(self.graph_connectivity, self.labels_per_agent, 'zero',
                                       'full', self.prob_sgd_dist_params, self.prob_aggr_dist_params, self.num_agents,
                                       self.cta, self.learning_rate, self.is_async)
        self.simulate_and_save_results(self.graph_connectivity, self.labels_per_agent, 'full',
                                       self.prob_sgd_type, self.prob_sgd_dist_params, self.prob_aggr_dist_params, self.num_agents,
                                       self.cta, self.learning_rate, not self.is_async)

    def graph_connectivity_experiment(self):
        self.simulation_count = 25
        self.simulation_counter = 1

        prob_aggr_types = [self.prob_aggr_type, 'full']
        prob_sgd_types = [self.prob_sgd_type, 'full']
        graph_connectivities = [1 / 5 * i for i in range(1, 6)]
        for prob_aggr_type in prob_aggr_types:
            for prob_sgd_type in prob_sgd_types:
                for graph_connectivity in graph_connectivities:
                    self.simulate_and_save_results(graph_connectivity, self.labels_per_agent, prob_aggr_type,
                                                   prob_sgd_type, self.prob_sgd_dist_params, self.prob_aggr_dist_params,
                                                   self.num_agents, self.cta, self.learning_rate, self.is_async)
        for graph_connectivity in graph_connectivities:
            self.simulate_and_save_results(graph_connectivity, self.labels_per_agent, 'zero',
                                           'full', self.prob_sgd_dist_params, self.prob_aggr_dist_params,
                                           self.num_agents, self.cta, self.learning_rate, self.is_async)

    def data_distribution_experiment(self):
        self.simulation_count = 50
        self.simulation_counter = 1

        prob_aggr_types = [self.prob_aggr_type, 'full']
        prob_sgd_types = [self.prob_sgd_type, 'full']
        labels_per_agents = [i for i in range(1, 11)]
        for prob_aggr_type in prob_aggr_types:
            for prob_sgd_type in prob_sgd_types:
                for labels_per_agent in labels_per_agents:
                    self.simulate_and_save_results(self.graph_connectivity, labels_per_agent, prob_aggr_type,
                                                   prob_sgd_type, self.prob_sgd_dist_params, self.prob_aggr_dist_params,
                                                   self.num_agents, self.cta, self.learning_rate, self.is_async)
        for labels_per_agent in labels_per_agents:
            self.simulate_and_save_results(self.graph_connectivity, labels_per_agent, 'zero',
                                           'full', self.prob_sgd_dist_params, self.prob_aggr_dist_params,
                                           self.num_agents, self.cta, self.learning_rate, self.is_async)

    def alpha_beta_experiment(self):
        self.simulation_count = 180
        self.simulation_counter = 1

        prob_aggr_types = [self.prob_aggr_type, 'full']
        prob_sgd_types = [self.prob_sgd_type, 'full']
        alphas = [0.5, 1, 2, 3, 4, 5]
        betas = [alpha for alpha in alphas]
        for alpha in alphas:
            for beta in betas:
                prob_dist_params = (alpha, beta)
                for prob_aggr_type in prob_aggr_types:
                    for prob_sgd_type in prob_sgd_types:
                        self.simulate_and_save_results(self.graph_connectivity, self.labels_per_agent,
                                                       prob_aggr_type, prob_sgd_type, prob_dist_params,
                                                       prob_dist_params, self.num_agents, self.cta, self.learning_rate, self.is_async)
                self.simulate_and_save_results(self.graph_connectivity, self.labels_per_agent,
                                               'zero', 'full', prob_dist_params, prob_dist_params, self.num_agents,
                                               self.cta, self.learning_rate, self.is_async)

    def alpha_beta_v2_experiment(self):
        self.simulation_count = 30
        self.simulation_counter = 1

        prob_aggr_types = [self.prob_aggr_type, 'full']
        prob_sgd_types = [self.prob_sgd_type, 'full']
        alpha_betas = [0.5, 0.6, 0.7, 0.8, 0.9, 1]
        for alpha_beta in alpha_betas:
            prob_dist_params = (alpha_beta, alpha_beta)
            for prob_aggr_type in prob_aggr_types:
                for prob_sgd_type in prob_sgd_types:
                    self.simulate_and_save_results(self.graph_connectivity, self.labels_per_agent,
                                                   prob_aggr_type, prob_sgd_type, prob_dist_params, prob_dist_params,
                                                   self.num_agents, self.cta, self.learning_rate, self.is_async)
            self.simulate_and_save_results(self.graph_connectivity, self.labels_per_agent,
                                           'zero', 'full', prob_dist_params, prob_dist_params, self.num_agents,
                                           self.cta, self.learning_rate, self.is_async)

    def alpha_beta_aggr_experiment(self):
        self.simulation_count = 30
        self.simulation_counter = 1

        prob_aggr_types = [self.prob_aggr_type, 'full']
        prob_sgd_types = [self.prob_sgd_type, 'full']
        alpha_betas_aggr = [0.5, 0.6, 0.7, 0.8, 0.9, 1]
        prob_sgd_dist_params = (0.5, 0.5) if self.dataset_name == 'FMNIST' else (0.8, 0.8)
        for alpha_beta in alpha_betas_aggr:
            prob_aggr_dist_params = (alpha_beta, alpha_beta)
            for prob_aggr_type in prob_aggr_types:
                for prob_sgd_type in prob_sgd_types:
                    self.simulate_and_save_results(self.graph_connectivity, self.labels_per_agent,
                                                   prob_aggr_type, prob_sgd_type, prob_sgd_dist_params,
                                                   prob_aggr_dist_params, self.num_agents, self.cta, self.learning_rate, self.is_async)
            self.simulate_and_save_results(self.graph_connectivity, self.labels_per_agent,
                                           'zero', 'full', prob_sgd_dist_params, prob_aggr_dist_params, self.num_agents,
                                           self.cta, self.learning_rate, self.is_async)

    def num_agents_experiment(self):
        self.simulation_count = 25
        self.simulation_counter = 1

        prob_aggr_types = [self.prob_aggr_type, 'full']
        prob_sgd_types = [self.prob_sgd_type, 'full']
        # num_agents_list = [10 * i for i in range(1,6)]
        num_agents_list = [2 * i for i in range(2, 6)]
        for num_agents in num_agents_list:
            for prob_aggr_type in prob_aggr_types:
                for prob_sgd_type in prob_sgd_types:
                    self.simulate_and_save_results(self.graph_connectivity, self.labels_per_agent, prob_aggr_type,
                                                   prob_sgd_type, self.prob_sgd_dist_params, self.prob_aggr_dist_params,
                                                   num_agents, self.cta, self.learning_rate, self.is_async)
            self.simulate_and_save_results(self.graph_connectivity, self.labels_per_agent, 'zero',
                                           'full', self.prob_sgd_dist_params, self.prob_aggr_dist_params, num_agents,
                                           self.cta, self.learning_rate, self.is_async)

    def dynamic_probs_experiment(self):
        self.simulation_count = 5
        self.simulation_counter = 1

        prob_aggr_types = [self.prob_aggr_type, 'full']
        prob_sgd_types = [self.prob_sgd_type, 'full']
        for prob_aggr_type in prob_aggr_types:
            for prob_sgd_type in prob_sgd_types:
                self.simulate_and_save_results(self.graph_connectivity, self.labels_per_agent, prob_aggr_type,
                                               prob_sgd_type, self.prob_sgd_dist_params, self.prob_aggr_dist_params,
                                               self.num_agents, self.cta, self.learning_rate, self.is_async)
        self.simulate_and_save_results(self.graph_connectivity, self.labels_per_agent, 'zero',
                                       'full', self.prob_sgd_dist_params, self.prob_aggr_dist_params, self.num_agents,
                                       self.cta, self.learning_rate, self.is_async)

    def truncnorm_experiment(self):
        # self.simulation_count = 25
        self.simulation_count = 30
        self.simulation_counter = 1

        prob_aggr_types = [self.prob_aggr_type, 'full']
        prob_sgd_types = [self.prob_sgd_type, 'full']
        # mean = 0.5
        means = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        sigma = 0.5
        # sigmas = [0.2, 0.4, 0.6, 0.8, 1.0]
        for mean in means:
            prob_dist_params = (mean, sigma)
            for prob_aggr_type in prob_aggr_types:
                for prob_sgd_type in prob_sgd_types:
                    self.simulate_and_save_results(self.graph_connectivity, self.labels_per_agent,
                                                   prob_aggr_type, prob_sgd_type, prob_dist_params, prob_dist_params,
                                                   self.num_agents, self.cta, self.learning_rate, self.is_async)
            self.simulate_and_save_results(self.graph_connectivity, self.labels_per_agent,
                                           'zero', 'full', prob_dist_params, prob_dist_params, self.num_agents,
                                           self.cta, self.learning_rate, self.is_async)

    def bimodal_experiment(self):
        self.simulation_count = 25
        self.simulation_counter = 1

        prob_aggr_types = [self.prob_aggr_type, 'full']
        prob_sgd_types = [self.prob_sgd_type, 'full']
        # mean = (0.2, 0.8)
        means = [(0, 1), (0.1, 0.9), (0.2, 0.8), (0.3, 0.7), (0.4, 0.6)]
        sigma = (0.1, 0.1)
        # sigmas = [(0.1, 0.1), (0.2, 0.2), (0.3, 0.3), (0.4, 0.4), (0.5, 0.5)]
        for mean in means:
        # for sigma in sigmas:
            prob_dist_params = ((mean[0], sigma[0]), (mean[1], sigma[1]))
            for prob_aggr_type in prob_aggr_types:
                for prob_sgd_type in prob_sgd_types:
                    self.simulate_and_save_results(self.graph_connectivity, self.labels_per_agent,
                                                   prob_aggr_type, prob_sgd_type, prob_dist_params, prob_dist_params,
                                                   self.num_agents, self.cta, self.learning_rate, self.is_async)
            self.simulate_and_save_results(self.graph_connectivity, self.labels_per_agent,
                                           'zero', 'full', prob_dist_params, prob_dist_params, self.num_agents,
                                           self.cta, self.learning_rate, self.is_async)

    def cta_experiment(self):
        self.simulation_count = 10
        self.simulation_counter = 1

        prob_aggr_types = [self.prob_aggr_type, 'full']
        prob_sgd_types = [self.prob_sgd_type, 'full']
        ctas = [False, True]
        for cta in ctas:
            for prob_aggr_type in prob_aggr_types:
                for prob_sgd_type in prob_sgd_types:
                    self.simulate_and_save_results(self.graph_connectivity, self.labels_per_agent,
                                                   prob_aggr_type, prob_sgd_type, self.prob_sgd_dist_params,
                                                   self.prob_aggr_dist_params, self.num_agents, cta, self.learning_rate, self.is_async)
            self.simulate_and_save_results(self.graph_connectivity, self.labels_per_agent,
                                           'zero', 'full', self.prob_sgd_dist_params, self.prob_aggr_dist_params,
                                           self.num_agents, cta, self.learning_rate, self.is_async)


    def learning_rate_experiment(self):
        self.simulation_count = 25
        self.simulation_counter = 1

        prob_aggr_types = [self.prob_aggr_type, 'full']
        prob_sgd_types = [self.prob_sgd_type, 'full']
        # num_agents_list = [10 * i for i in range(1,6)]
        learning_rate_list = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]
        for learning_rate in learning_rate_list:
            for prob_aggr_type in prob_aggr_types:
                for prob_sgd_type in prob_sgd_types:
                    self.simulate_and_save_results(self.graph_connectivity, self.labels_per_agent, prob_aggr_type,
                                                   prob_sgd_type, self.prob_sgd_dist_params, self.prob_aggr_dist_params,
                                                   self.num_agents, self.cta, learning_rate, self.is_async)
            self.simulate_and_save_results(self.graph_connectivity, self.labels_per_agent, 'zero',
                                           'full', self.prob_sgd_dist_params, self.prob_aggr_dist_params, self.num_agents,
                                           self.cta, learning_rate, self.is_async)

    def simulate_and_save_results(self, graph_connectivity, labels_per_agent, prob_aggr_type, prob_sgd_type,
                                  prob_sgd_dist_params, prob_aggr_dist_params, num_agents, cta, learning_rate, is_async):
        filepath = self.file_info(graph_connectivity, labels_per_agent, prob_aggr_type, prob_sgd_type,
                                  prob_sgd_dist_params, prob_aggr_dist_params, num_agents, cta, learning_rate, is_async)

        print(f"Simulation {self.simulation_counter}/{self.simulation_count}: {filepath}")
        self.simulation_counter += 1
        if os.path.exists(filepath):
            return

        self.dSpodFL.reset(
            graph_connectivity=graph_connectivity,
            labels_per_agent=labels_per_agent,
            prob_aggr_type=prob_aggr_type,
            prob_sgd_type=prob_sgd_type,
            sim_type=self.sim_type,
            prob_sgd_dist_params=prob_sgd_dist_params,
            prob_aggr_dist_params=prob_aggr_dist_params,
            num_agents=num_agents,
            learning_rate=learning_rate,
            is_async=is_async
        )
        log = self.dSpodFL.run()
        utils.save_results(log, filepath)

    def file_info(self, graph_connectivity, labels_per_agent, prob_aggr_type, prob_sgd_type, prob_sgd_dist_params,
                  prob_aggr_dist_params, num_agents, cta, learning_rate, is_async):
        dirname = f"../results/{self.seed}"
        dirpath = os.path.join(os.getcwd(), dirname)
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)

        filename = f"{self.model_name}_{self.dataset_name}_{self.num_epochs}epochs_{self.num_agents}m"
        filename += f"_{graph_connectivity}conn_{labels_per_agent}labels_{prob_aggr_type}aggr_{prob_sgd_type}sgd"
        filename += f"_{prob_sgd_dist_params}_{prob_aggr_dist_params}_{num_agents}agents_{self.sim_type}"
        filename += f"_{cta}cta_{learning_rate}lr_{is_async}async_{self.seed}seed.xlsx"

        filepath = os.path.join(dirpath, filename)
        return filepath


def main():
    seed = 42    # 1
    # seed = 43    # 2
    # seed = 44    # 3
    # seed = 45    # 4
    # seed = 46    # 5
    # seed = 47    # 6
    # seed = 48    # 7
    random.seed(seed)
    sims = Simulations(seed)
    sims.run()


if __name__ == '__main__':
    main()
