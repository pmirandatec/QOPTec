import os
import numpy as np
import pandas as pd
from time import time
from pathlib import Path

import dimod
from dwave.optimization.generators import knapsack
from dwave.system.composites import EmbeddingComposite
from dwave.system.samplers import DWaveSampler
from dwave.system import LeapHybridBQMSampler, LeapHybridCQMSampler, LeapHybridNLSampler
from qiskit_optimization.applications import Knapsack
from qiskit_optimization.converters import QuadraticProgramToQubo

import utils

class KnapsackSolver:
    def __init__(self, instance, sapi_token):
        self.instance = instance
        self.sapi_token = sapi_token
        self.solver = None
        self.best_fitness = None
        self.mean_fitness = None
        self.median_fitness = None
        self.best_solution = None
        self.number_of_solutions = None
        self.unique_solutions = None
        self.best_sol_ocurrence =None
        self.best_energy = None
        self.mean_energy = None
        self.median_energy = None
        self.total_weight = None
        self.n_items = None

        # Times
        self.runtime = None
        self.solver_time = None
        self.qpu_time = None
        self.dwave_time = None
        
        # Problem initialization
        self.values, self.weights, self.max_weight = utils.parse_kp_format(self.instance)

    ########################################################################################################################
    # QPU SOLVER
    ########################################################################################################################

    def solve_with_qpu(self):
        start_time = time()

        # Initialize problem data and sampler
        knapsack = Knapsack(self.values, self.weights, self.max_weight)
        qp = knapsack.to_quadratic_program()
        qp2qubo = QuadraticProgramToQubo()
        qubo = qp2qubo.convert(qp)
        bqm_binary = dimod.as_bqm(qubo.objective.linear.to_array(), qubo.objective.quadratic.to_array(),
                                  dimod.BINARY)
        sampler = EmbeddingComposite(DWaveSampler(region="na-west-1", token=self.sapi_token))
        self.solver = 'qpu'

        # Execute solver and extract metrics
        solver_time_start = time()
        res = sampler.sample(bqm_binary, num_reads=1000, label='Knapsack_QPU')
        self.solver_time = time() - solver_time_start
        self.dwave_time = res.info['timing']['run_time'] / 1000000
        self.qpu_time = float(res.info['timing']['qpu_access_time']) / 1000000
        self.number_of_solutions = len(res.data_vectors['energy'])
        self.unique_solutions = len(np.unique(res.data_vectors['energy']))
        self.mean_energy = np.mean(res.data_vectors['energy'])
        self.median_energy = np.median(np.sort(res.data_vectors['energy']))
        sample_list_aux = res.record.sample
        solution_list = []
        fitness_list = []
        total_weight_list = []

        # Compute the fitness and weight of every solution
        for sample in sample_list_aux:
            x = np.array(sample)
            z = knapsack.interpret(x[0:len(self.weights) + (len(self.weights) * len(self.weights))])
            solution_list.append(z)
            fitness = 0
            total_weight = 0
            for i in z:
                if i < len(self.values):
                    fitness = fitness + self.values[i]
                    total_weight = total_weight + self.weights[i]
            fitness_list.append(fitness)
            total_weight_list.append(total_weight)

        self.best_fitness, self.total_weight, index = utils.get_max_value_under_weight(fitness_list, total_weight_list,
                                                                             self.max_weight)
        self.best_energy = res.data_vectors['energy'][index]
        self.best_sol_ocurrence = np.count_nonzero(res.data_vectors['energy'] == self.best_energy)
        best_sample = res.record[index][0]
        best_solution_aux = knapsack.interpret(best_sample[0:len(self.weights) + (len(self.weights) * len(self.weights))])
        self.best_solution = [item for item in best_solution_aux if item <= len(self.values)]
        self.mean_fitness = float(np.mean(fitness_list))
        self.median_fitness = float(np.median(fitness_list))

        self.runtime = time() - start_time

    ########################################################################################################################
    # BQM SOLVER
    ########################################################################################################################

    def solve_with_bqm(self):


        # Initialize problem data and sampler
        knapsack = Knapsack(self.values, self.weights, self.max_weight)
        qp = knapsack.to_quadratic_program()
        qp2qubo = QuadraticProgramToQubo()
        qubo = qp2qubo.convert(qp)
        bqm_binary = dimod.as_bqm(qubo.objective.linear.to_array(), qubo.objective.quadratic.to_array(), dimod.BINARY)
        sampler = LeapHybridBQMSampler(token=self.sapi_token)
        self.solver = 'bqm'

        # Execute solver and extract metrics
        solver_time_start = time()
        res = sampler.sample(bqm_binary, label='Knapsack_BQM')
        self.solver_time = time() - solver_time_start
        self.dwave_time = float(res.info.get('run_time')) / 1000000
        self.qpu_time = float(res.info.get('qpu_access_time')) / 1000000
        self.number_of_solutions = len(res.data_vectors['energy'])
        self.unique_solutions = len(np.unique(res.data_vectors['energy']))
        self.mean_energy = np.mean(res.data_vectors['energy'])
        self.median_energy = np.median(np.sort(res.data_vectors['energy']))
        sample_list_aux = res.record.sample
        solution_list = []
        fitness_list = []
        total_weight_list = []

        # Compute the fitness and weight of every solution
        for sample in sample_list_aux:
            x = np.array(sample)
            z = knapsack.interpret(x[0:len(self.weights) + (len(self.weights) * len(self.weights))])
            solution_list.append(z)
            fitness = 0
            total_weight = 0
            for i in z:
                if i < len(self.values):
                    fitness = fitness + self.values[i]
                    total_weight = total_weight + self.weights[i]
            fitness_list.append(fitness)
            total_weight_list.append(total_weight)

        self.best_fitness, self.total_weight, index = utils.get_max_value_under_weight(fitness_list, total_weight_list, self.max_weight)
        self.best_energy = res.data_vectors['energy'][index]
        self.best_sol_ocurrence = np.count_nonzero(res.data_vectors['energy'] == self.best_energy)
        best_sample = res.record[index][0]
        best_solution_aux = knapsack.interpret(best_sample[0:len(self.weights) + (len(self.weights) * len(self.weights))])
        self.best_solution = [item for item in best_solution_aux if item <= len(self.values)]
        self.mean_fitness = float(np.mean(fitness_list))
        self.median_fitness = float(np.median(fitness_list))
        self. runtime = time() - solver_time_start

    ########################################################################################################################
    # CQM SOLVER
    ########################################################################################################################

    def solve_with_cqm(self):
        time_start = time()

        # Initialize problem data and sampler
        num_items = len(self.values)
        cqm = dimod.ConstrainedQuadraticModel()
        obj = dimod.BinaryQuadraticModel(vartype='BINARY')
        constraint = dimod.QuadraticModel()
        for i in range(num_items):
            obj.add_variable(i)
            obj.set_linear(i, - self.values[i])
            constraint.add_variable('BINARY', i)
            constraint.set_linear(i, self.weights[i])
        cqm.set_objective(obj)
        cqm.add_constraint(constraint, sense="<=", rhs=self.max_weight, label='capacity')
        sampler = LeapHybridCQMSampler(token=self.sapi_token)
        self.solver = 'cqm'

        # Execute solver and extract metrics
        solver_time_start = time()
        sampleset = sampler.sample_cqm(cqm, label='Knapsack_CQM')
        feasible_sampleset = sampleset.filter(lambda row: row.is_feasible)
        self.solver_time = time() - solver_time_start
        self.dwave_time = float(sampleset.info['run_time'] / 1000000)
        self.qpu_time = float(sampleset.info['qpu_access_time']) / 1000000
        self.mean_fitness = - float(np.mean(feasible_sampleset.data_vectors['energy']))
        self.median_fitness = - float(np.median(feasible_sampleset.data_vectors['energy']))
        self.unique_solutions = len(np.unique(feasible_sampleset.data_vectors['energy']))
        self.best_fitness = - float(np.min(feasible_sampleset.data_vectors['energy']))
        self.number_of_solutions = len(feasible_sampleset.data_vectors['energy'])
        self.best_sol_ocurrence = np.count_nonzero(feasible_sampleset.data_vectors['energy'] == - self.best_fitness)

        # Compute the fitness and weight of every solution
        if not len(feasible_sampleset):
            raise ValueError("No feasible solution found")

        best = feasible_sampleset.first
        self. best_solution = [key for key, val in best.sample.items() if val == 1.0]
        selected_weights = [self.weights[i] for i in self.best_solution]
        self.total_weight = sum(selected_weights)
        self.best_energy = - self.best_fitness
        self.mean_energy = - self.mean_fitness
        self.median_energy = - self.median_fitness
        self.runtime = time() - time_start

    ########################################################################################################################
    # NL SOLVER
    ########################################################################################################################

    def solve_with_nl(self):
        start_time = time()

        # Initialize problem data and sampler
        kp_nl = knapsack(self.values, self.weights, self.max_weight)
        dwave_url = 'https://cloud.dwavesys.com/sapi'
        sampler = LeapHybridNLSampler(token=self.sapi_token, endpoint=dwave_url)
        self.solver = 'nl'

        # Execute solver and extract metrics
        time_in_solver_start = time()
        futures = sampler.sample(kp_nl, time_limit=5, label="Knapsack_NL")
        self.solver_time = time() - time_in_solver_start
        self.dwave_time = float(futures.result().info['timing']['run_time'] / 1000000)
        self.qpu_time = float(futures.result().info['timing']['qpu_access_time']) / 1000000
        self.best_energy = None
        energies = []
        solution_list = []
        for i in range(kp_nl.states.size()):
            sol = next(kp_nl.iter_decisions()).state(i).astype(int)
            solver_objective = kp_nl.objective.state(i)
            energies.append(solver_objective)
            solution_list.append(sol)
            if best_energy == None:
                best_energy = int(solver_objective)
        self.best_fitness = - float(min(energies))
        self.mean_fitness = - float(np.mean(energies))
        self.median_fitness = - float(np.median(energies))
        self.unique_solutions = len(np.unique(energies))
        self.number_of_solutions = len(energies)
        self.best_solution = solution_list[energies.index(-self.best_fitness)]
        self.best_sol_ocurrence = energies.count(-self.best_fitness)
        selected_weights = [self.weights[i] for i in sol]
        self.total_weight = sum(selected_weights)
        self.best_energy = - self.best_fitness
        self.mean_energy = - self.mean_fitness
        self.median_energy = - self.median_fitness
        self.runtime = time() - start_time

    ########################################################################################################################
    # GENERAL METHOD
    ########################################################################################################################

    def solve(self, solver):
        solver_mapping = {
            "qpu": self.solve_with_qpu,
            "bqm": self.solve_with_bqm,
            "cqm": self.solve_with_cqm,
            "nl": self.solve_with_nl,
        }

        if solver in solver_mapping:
            solver_mapping[solver]()
        else:
            raise ValueError(f"Invalid method '{solver}'. Use qpu, bqm, or cqm")


    ########################################################################################################################
    # DATAFRAME GENERATION
    ########################################################################################################################

    def get_dataframe(self):

        # Get problem data
        problem_info_list = utils.parse_kp_format(self.instance)
        self.n_items = len(problem_info_list[0])

        # Create dataframe with instance analysis
        filename = 'knapsack.xlsx'
        subfolder = '../outputs/kp'
        filepath = os.path.join(subfolder, filename)
        if os.path.exists(filepath):
            df = pd.read_excel(filepath)
        else:
            df = pd.DataFrame(columns=['Instance', 'Solver', 'Instance size', 'Best fitness',
                                       'Mean fitness', 'Median fitness',
                                       'Best solution', 'Number of solutions', 'Unique fitness',
                                       'Best fitness ocurrences', 'Execution time', 'DWave time', 'Time in solver',
                                       'QPU access time', 'Best energy', 'Mean energy', 'Median energy',
                                       'Total weight'])

        new_row = {'Instance': Path(self.instance).name, 'Solver': self.solver, 'Instance size': self.n_items, 'Best fitness': self.best_fitness,
                   'Mean fitness': self.mean_fitness, 'Median fitness': self.median_fitness, 'Best solution': self.best_solution,
                   'Number of solutions': self.number_of_solutions, 'Unique fitness': self.unique_solutions,
                   'Best fitness ocurrences': self.best_sol_ocurrence, 'Execution time': self.runtime, 'DWave time': self.dwave_time,
                   'Time in solver': self.solver_time, 'QPU access time': self.qpu_time, 'Best energy': self.best_energy,
                   'Mean energy': self.mean_energy, 'Median energy': self.median_energy, 'Total weight': self.total_weight}

        new_df = pd.DataFrame([new_row])
        df = pd.concat([df, new_df], ignore_index=True)
        df.to_excel(filepath, index=False)
        print('Dataframe for instance', self.instance, 'has been generated')
        return df