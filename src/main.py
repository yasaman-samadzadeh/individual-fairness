from __future__ import annotations

from functools import partial
import os
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
from ConfigSpace import (
    Categorical,
    Configuration,
    ConfigurationSpace,
    EqualsCondition,
    Float,
    InCondition,
    Integer,
)

from fairlearn import metrics


# Sklear utils
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_validate

# Classification algorithms
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression

# SMAC stuff
from smac import HyperparameterOptimizationFacade as HPOFacade
from smac import Scenario
from smac.facade.abstract_facade import AbstractFacade
from smac.multi_objective.parego import ParEGO

__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"

from utils.datasets import load_dataset_from_openml


class DynamicPipeline:

    X = None
    y = None
    categorical_indicator = []
    sensitive_indicator = []
    feature_names = []
    standard_obj = None
    fairness_obj = None

    def __init__(self, standard_obj, fairness_obj):
        X, y, categorical_indicator, sensitive_indicator, feature_names = (
            load_dataset_from_openml(31, "8")
        )
        self.X = X
        self.y = y
        self.categorical_indicator = categorical_indicator
        self.sensitive_indicator = sensitive_indicator
        self.feature_names = feature_names
        self.standard_obj = standard_obj
        self.fairness_obj = fairness_obj

    @property
    def configspace(self) -> ConfigurationSpace:
        cs = ConfigurationSpace()

        algorithm_type = Categorical(
            "type", ["RandomForestClassifier", "MLPClassifier"], default="RandomForestClassifier")

        criterion = Categorical(
            "criterion", ["gini", "entropy", "log_loss"], default="gini")
        max_depth = Integer("max_depth", (5, 10), default=5)

        n_layer = Integer("n_hidden_layers", (1, 5), default=1)
        n_neurons = Integer("n_neurons", (8, 256), log=True, default=10)
        activation = Categorical(
            "activation", ["logistic", "tanh", "relu"], default="tanh")
        solver = Categorical(
            "solver", ["lbfgs", "sgd", "adam"], default="adam")

        cs.add_hyperparameters(
            [algorithm_type, criterion, max_depth, n_layer, n_neurons, activation, solver])

        use_criterion = EqualsCondition(
            child=criterion, parent=algorithm_type, value="RandomForestClassifier")
        use_max_depth = EqualsCondition(
            child=max_depth, parent=algorithm_type, value="RandomForestClassifier")
        use_n_layer = EqualsCondition(
            child=n_layer, parent=algorithm_type, value="MLPClassifier")
        use_n_neurons = EqualsCondition(
            child=n_neurons, parent=algorithm_type, value="MLPClassifier")
        use_activation = EqualsCondition(
            child=activation, parent=algorithm_type, value="MLPClassifier")
        use_solver = EqualsCondition(
            child=solver, parent=algorithm_type, value="MLPClassifier")

        # We can also add multiple conditions on hyperparameters at once:
        cs.add_conditions([use_criterion, use_max_depth, use_n_layer,
                          use_n_layer, use_n_neurons, use_activation, use_solver])

        return cs

    def train(self, config: Configuration, seed: int = 0) -> dict[str, float]:

        algo_hyperparams = {
            hyperparam: config[hyperparam]
            for hyperparam in config
            if hyperparam != "type"
        }

        if config["type"] == "MLPClassifier":
            algo_hyperparams["hidden_layer_sizes"] = (
                algo_hyperparams["n_neurons"]) * algo_hyperparams["n_hidden_layers"]
            algo_hyperparams.pop("n_neurons", None)
            algo_hyperparams.pop("n_hidden_layers", None)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")

            classifier = globals()[config["type"]](**algo_hyperparams)
            # Returns the 5-fold cross validation accuracy
            # to make CV splits consistent
            cv = StratifiedKFold(n_splits=5)

            standard_scores = cross_validate(
                classifier,
                self.X.copy(),
                self.y.copy(),
                scoring=[self.standard_obj],
                cv=cv.split(self.X, self.y),
                return_estimator=True,
                return_train_score=False,
                # return_indices=True,
                verbose=0,
            )

        metrics_module = globals()["metrics"]
        performance_metric = getattr(metrics_module, self.fairness_obj)
        fair_scores = []

        for fold, (train_indeces, test_indeces) in enumerate(cv.split(self.X, self.y)):
            # test_indeces = scores["indices"]["test"][fold]
            x_original = self.X.copy()[test_indeces, :]
            sensitive_mask = [i for i, x in enumerate(
                self.sensitive_indicator) if x == True]
            x_sensitive = x_original[:, sensitive_mask]

            # forse fare .reshape(-1, 1) in caso di intersectionality
            x_sensitive = (
                x_sensitive if len(
                    sensitive_mask) > 1 else x_sensitive.reshape(-1)
            )
            fair_scores += [
                performance_metric(
                    y_true=np.array(self.y.copy()[test_indeces]),
                    y_pred=np.array(standard_scores["estimator"]
                                    [fold].predict(x_original)),
                    sensitive_features=x_sensitive,
                )
            ]

        return {
            self.standard_obj: 1 - np.mean(standard_scores["test_" + self.standard_obj]),
            self.fairness_obj: np.mean(fair_scores),
        }


def get_pareto_front(smac: AbstractFacade) -> tuple[list[Configuration], list[list[float]]]:
    """Returns the Pareto front of the runhistory.

    Returns
    -------
    configs : list[Configuration]
        The configs of the Pareto front.
    costs : list[list[float]]
        The costs from the configs of the Pareto front.
    """

    # Get costs from runhistory first
    average_costs = []
    configs = smac.runhistory.get_configs()
    for config in configs:
        # Since we use multiple seeds, we have to average them to get only one cost value pair for each
        # configuration
        # Luckily, SMAC already does this for us
        average_cost = smac.runhistory.average_cost(config)
        average_costs += [average_cost]

    # Let's work with a numpy array
    costs = np.vstack(average_costs)

    is_efficient = np.arange(costs.shape[0])
    next_point_index = 0  # Next index in the is_efficient array to search for
    while next_point_index < len(costs):
        nondominated_point_mask = np.any(
            costs < costs[next_point_index], axis=1)
        nondominated_point_mask[next_point_index] = True
        # Remove dominated points
        is_efficient = is_efficient[nondominated_point_mask]
        costs = costs[nondominated_point_mask]
        next_point_index = np.sum(
            nondominated_point_mask[:next_point_index]) + 1

    return [configs[i] for i in is_efficient], [average_costs[i] for i in is_efficient]


def plot_pareto(smac: AbstractFacade, incumbents) -> None:
    """Plots configurations from SMAC and highlights the best configurations in a Pareto front."""
    # Get Pareto costs
    # print([smac.runhistory.get_cost(incumbent) for incumbent in incumbents])
    _, c = get_pareto_front(smac)
    pareto_costs = np.array(c)

    # Sort them a bit
    pareto_costs = pareto_costs[pareto_costs[:, 0].argsort()]

    # Get all other costs from runhistory
    average_costs = []
    for config in smac.runhistory.get_configs():
        # Since we use multiple seeds, we have to average them to get only one cost value pair for each configuration
        average_cost = smac.runhistory.average_cost(config)

        if average_cost not in c:
            average_costs += [average_cost]

    # Let's work with a numpy array
    costs = np.vstack(average_costs)
    costs_x, costs_y = costs[:, 0], costs[:, 1]
    pareto_costs_x, pareto_costs_y = pareto_costs[:, 0], pareto_costs[:, 1]

    plt.scatter(costs_x, costs_y, marker="x")
    plt.scatter(pareto_costs_x, pareto_costs_y, marker="x", c="r")
    plt.step(
        [pareto_costs_x[0]] + pareto_costs_x.tolist() + [np.max(costs_x)
                                                         ],  # We add bounds
        [np.max(costs_y)] + pareto_costs_y.tolist() + \
        [np.min(pareto_costs_y)],  # We add bounds
        where="post",
        linestyle=":",
    )

    plt.title("Pareto-Front")
    plt.xlabel(smac.scenario.objectives[0])
    plt.ylabel(smac.scenario.objectives[1])
    # plt.show()
    plt.gcf().savefig(os.path.join("plots", "toy_example.png"))


if __name__ == "__main__":

    np.random.seed(42)

    objectives = ["balanced_accuracy", "demographic_parity_difference"]

    mlp = DynamicPipeline(standard_obj=objectives[0],
                          fairness_obj=objectives[1])

    # Define our environment variables
    scenario = Scenario(
        mlp.configspace,
        objectives=objectives,
        walltime_limit=60,  # After 40 seconds, we stop the hyperparameter optimization
        n_trials=200,  # Evaluate max 200 different trials
        n_workers=1,
    )

    # We want to run five random configurations before starting the optimization.
    initial_design = HPOFacade.get_initial_design(scenario, n_configs=5)
    multi_objective_algorithm = ParEGO(scenario)

    # Create our SMAC object and pass the scenario and the train method
    smac = HPOFacade(
        scenario,
        mlp.train,
        initial_design=initial_design,
        multi_objective_algorithm=multi_objective_algorithm,
        overwrite=True,
    )

    # Let's optimize
    # Keep in mind: The incumbent is ambiguous here because of ParEGO
    incumbents = smac.optimize()

    # Get cost of default configuration
    default_cost = smac.validate(mlp.configspace.get_default_configuration())
    print(f"Default costs: {default_cost}\n")

    # print("Validated costs from the Pareto front:")
    # for i, config in enumerate(smac.runhistory.get_pareto_front()[0]):
    #     cost = smac.validate(config)
    #     print(cost)

    # Let's plot a pareto front
    plot_pareto(smac, incumbents)
