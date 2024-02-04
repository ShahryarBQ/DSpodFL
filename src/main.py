import os
import random

import utils
from DSpodFL import DSpodFL


class Simulations:
    def __init__(self, seed):
        self.model_name = 'SVM'
        self.dataset_name = 'FMNIST'
        self.num_epochs = 2
        self.num_agents = 10
        self.graph_connectivity = 0.4
        self.labels_per_agent = 1
        batch_size = 16
        learning_rate = 0.01
        self.prob_aggr_type = 'beta'
        self.prob_sgd_type = 'beta'
        self.sim_type = 'data_dist'
        self.prob_dist_params = (0.5, 0.5)    # (alpha, beta) or (min, max)
        termination_delay = 500
        DandB = (None,1)
        self.seed = seed

        self.simulation_count = None
        self.simulation_counter = None

        self.dSpodFL = DSpodFL(
            self.model_name,
            self.dataset_name,
            self.num_epochs,
            self.num_agents,
            self.graph_connectivity,
            self.labels_per_agent,
            batch_size,
            learning_rate,
            self.prob_aggr_type,
            self.prob_sgd_type,
            self.sim_type,
            self.prob_dist_params,
            termination_delay,
            DandB
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
            self.alpha_beta_v2_experiment()
        elif self.sim_type == 'num_agents':
            self.num_agents_experiment()

    def efficiency_experiment(self):
        self.simulation_count = 5
        self.simulation_counter = 1

        prob_aggr_types = [self.prob_aggr_type, 'full']
        prob_sgd_types = [self.prob_sgd_type, 'full']
        for prob_aggr_type in prob_aggr_types:
            for prob_sgd_type in prob_sgd_types:
                self.simulate_and_save_results(self.graph_connectivity, self.labels_per_agent, prob_aggr_type,
                                               prob_sgd_type, self.prob_dist_params, self.num_agents)
        self.simulate_and_save_results(self.graph_connectivity, self.labels_per_agent, 'zero',
                                       'full', self.prob_dist_params, self.num_agents)

    def graph_connectivity_experiment(self):
        self.simulation_count = 25
        self.simulation_counter = 1

        prob_aggr_types = [self.prob_aggr_type, 'full']
        prob_sgd_types = [self.prob_sgd_type, 'full']
        graph_connectivities = [1/5 * i for i in range(1, 6)]
        for prob_aggr_type in prob_aggr_types:
            for prob_sgd_type in prob_sgd_types:
                for graph_connectivity in graph_connectivities:
                    self.simulate_and_save_results(graph_connectivity, self.labels_per_agent, prob_aggr_type,
                                                   prob_sgd_type, self.prob_dist_params, self.num_agents)
        for graph_connectivity in graph_connectivities:
            self.simulate_and_save_results(graph_connectivity, self.labels_per_agent, 'zero',
                                           'full', self.prob_dist_params, self.num_agents)

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
                                                   prob_sgd_type, self.prob_dist_params, self.num_agents)
        for labels_per_agent in labels_per_agents:
            self.simulate_and_save_results(self.graph_connectivity, labels_per_agent, 'zero',
                                           'full', self.prob_dist_params, self.num_agents)


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
                                                       prob_aggr_type, prob_sgd_type, prob_dist_params, self.num_agents)
                self.simulate_and_save_results(self.graph_connectivity, self.labels_per_agent,
                                               'zero', 'full', prob_dist_params, self.num_agents)


    def alpha_beta_v2_experiment(self):
        self.simulation_count = 25
        self.simulation_counter = 1

        prob_aggr_types = [self.prob_aggr_type, 'full']
        prob_sgd_types = [self.prob_sgd_type, 'full']
        alpha_betas = [0.1, 0.3, 0.5, 0.7, 0.9]
        for alpha_beta in alpha_betas:
            prob_dist_params = (alpha_beta, alpha_beta)
            for prob_aggr_type in prob_aggr_types:
                for prob_sgd_type in prob_sgd_types:
                    self.simulate_and_save_results(self.graph_connectivity, self.labels_per_agent,
                                                   prob_aggr_type, prob_sgd_type, prob_dist_params, self.num_agents)
            self.simulate_and_save_results(self.graph_connectivity, self.labels_per_agent,
                                           'zero', 'full', prob_dist_params, self.num_agents)


    def num_agents_experiment(self):
        self.simulation_count = 25;
        self.simulation_counter = 1;

        prob_aggr_types = [self.prob_aggr_type, 'full']
        prob_sgd_types = [self.prob_sgd_type, 'full']
        num_agents_list = [10 * i for i in range(1,6)]
        for num_agents in num_agents_list:
            for prob_aggr_type in prob_aggr_types:
                for prob_sgd_type in prob_sgd_types:
                    self.simulate_and_save_results(self.graph_connectivity, self.labels_per_agent, prob_aggr_type,
                                                   prob_sgd_type, self.prob_dist_params, num_agents)
            self.simulate_and_save_results(self.graph_connectivity, self.labels_per_agent, 'zero',
                                           'full', self.prob_dist_params, num_agents)


    def simulate_and_save_results(self, graph_connectivity, labels_per_agent, prob_aggr_type, prob_sgd_type, prob_dist_params, num_agents):
        filepath = self.file_info(graph_connectivity, labels_per_agent, prob_aggr_type, prob_sgd_type, prob_dist_params, num_agents)

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
            prob_dist_params=prob_dist_params,
            num_agents=num_agents
        )
        log = self.dSpodFL.run()
        utils.save_results(log, filepath)

    def file_info(self, graph_connectivity, labels_per_agent, prob_aggr_type, prob_sgd_type, prob_dist_params, num_agents):
        dirname = "../results"
        dirpath = os.path.join(os.getcwd(), dirname)
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)

        filename = f"{self.model_name}_{self.dataset_name}_{self.num_epochs}epochs"
        filename += f"_{self.num_agents}m_{graph_connectivity}conn_{labels_per_agent}labels"
        filename += f"_{prob_aggr_type}aggr_{prob_sgd_type}sgd_{prob_dist_params}"
        filename += f"_{num_agents}agents_{self.sim_type}_{self.seed}seed.xlsx"

        filepath = os.path.join(dirpath, filename)
        return filepath


def main():
    seed = 42    # 1
    # seed = 43    # 2
    # seed = 44    # 3
    # seed = 45    # 4
    # seed = 46    # 5
    # seed = 47    # 6
    random.seed(seed)
    sims = Simulations(seed)
    sims.run()


if __name__ == '__main__':
    main()
