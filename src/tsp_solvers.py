import networkx as nx
import numpy as np
import os
import pandas as pd
from time import time
from pathlib import Path

import dimod
from dwave.optimization.generators import traveling_salesperson
from qiskit_optimization.converters import QuadraticProgramToQubo
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite
from dwave.system import LeapHybridBQMSampler, LeapHybridCQMSampler, LeapHybridNLSampler

import utils

class TravellingSalesmanSolver:
    def __init__(self, instance, sapi_token):
        self.instance = instance
        self.sapi_token = sapi_token
        self.solver = None
        self.best_distance = None
        self.mean_distance = None
        self.median_distance = None
        self.best_solution = None
        self.number_of_solutions = None
        self.unique_solutions = None
        self.best_sol_ocurrence = None
        self.best_energy = None
        self.mean_energy = None
        self.median_energy = None

        # Times
        self.runtime = None
        self.solver_time = None
        self.qpu_time = None
        self.dwave_time = None

        # Problem initialization
        self.tsp = utils.parse_tsplib_format(self.instance)
        self.adj_matrix = nx.to_numpy_array(self.tsp.graph)

    ########################################################################################################################
    # QPU SOLVER
    ########################################################################################################################

    def solve_with_qpu(self):
        start_time = time()
        qp = self.tsp.to_quadratic_program()
        qp2qubo = QuadraticProgramToQubo()
        qubo = qp2qubo.convert(qp)
        bqm_binary = dimod.as_bqm(qubo.objective.linear.to_array(), qubo.objective.quadratic.to_array(),
                                  dimod.BINARY)
        self.solver = 'qpu'

        sampler = EmbeddingComposite(DWaveSampler(region="na-west-1", token=self.sapi_token))
        solver_time_start = time()
        sample_set = sampler.sample(bqm_binary, num_reads=1000, label='test_SoftwareX')

        self.solver_time = time() - solver_time_start
        self.dwave_time = sample_set.info['timing']['qpu_access_time'] / 1000000
        self.qpu_time = float(sample_set.info['timing']['qpu_sampling_time'] ) / 1000000
        self.number_of_solutions = len(sample_set.data_vectors['energy'])
        self.unique_solutions = len(np.unique(sample_set.data_vectors['energy']))
        self.mean_energy = np.mean(sample_set.data_vectors['energy'])
        self.median_energy = np.median(np.sort(sample_set.data_vectors['energy']))
        sample_list_aux = sample_set.record.sample
        solution_list = []
        distance_list = []

        # We compute the distance of every solution
        for sample in sample_list_aux:
            z = self.tsp.interpret(sample)
            solution_list.append(z)
        solution_list = utils.filter_unique_integer_sublists(solution_list)  # This removes unfeasible solutions
        for solution in solution_list:
            distance = self.tsp.tsp_value(solution, self.adj_matrix)
            distance_list.append(distance)
        self.best_distance = min(distance_list)
        index = distance_list.index(self.best_distance)
        self.best_solution = solution_list[index]
        self.best_energy = sample_set.record[0][1]  # Best energy. May (probably will correspond to unfeasible
        self.best_sol_ocurrence = np.count_nonzero(distance_list == self.best_distance)
        self.mean_distance = np.mean(distance_list)
        self.median_distance = np.median(distance_list)
        self.runtime = time() - start_time

    ########################################################################################################################
    # BQM SOLVER
    ########################################################################################################################

    def solve_with_bqm(self):
        start_time = time()

        # Initialize problem data and sampler
        qp = self.tsp.to_quadratic_program()
        qp2qubo = QuadraticProgramToQubo()
        qubo = qp2qubo.convert(qp)
        bqm_binary = dimod.as_bqm(qubo.objective.linear.to_array(), qubo.objective.quadratic.to_array(), dimod.BINARY)
        sampler = LeapHybridBQMSampler(token=self.sapi_token)
        self.solver = 'bqm'

        # Execute solver and extract metrics
        solver_time_start = time()
        sample_set = sampler.sample(bqm_binary, label='test_SoftwareX')
        solver_time_end = time()
        self.solver_time = solver_time_end - solver_time_start
        self.dwave_time = sample_set.info.get('run_time') / 1000000
        self.qpu_time = float(sample_set.info.get('qpu_access_time') / 1000000)
        self.number_of_solutions = len(sample_set.data_vectors['energy'])
        self.unique_energies = len(np.unique(sample_set.data_vectors['energy']))
        self.mean_energy = np.mean(sample_set.data_vectors['energy'])
        self.median_energy = np.median(np.sort(sample_set.data_vectors['energy']))
        sample_list_aux = sample_set.record.sample
        solution_list = []
        distance_list = []

        # Compute the distance of every solution
        for sample in sample_list_aux:
            z = self.tsp.interpret(sample)
            solution_list.append(z)
        solution_list = utils.filter_unique_integer_sublists(solution_list)
        for solution in solution_list:
            distance = self.tsp.tsp_value(solution, self.adj_matrix)
            distance_list.append(distance)
        self.best_distance = min(distance_list)
        index = distance_list.index(self.best_distance)
        self.best_solution = solution_list[index]
        self.best_energy = sample_set.record[0][1]
        self.best_sol_ocurrence = np.count_nonzero(distance_list == self.best_distance)
        self.mean_distance = np.mean(distance_list)
        self.median_distance = np.median(distance_list)

        self.runtime = time() - start_time


########################################################################################################################
# CQM SOLVER
########################################################################################################################

    def solve_with_cqm(self):
        start_time = time()

        # Initialize problem data and sampler
        cqm = dimod.ConstrainedQuadraticModel()
        x = [[None] * (len(self.adj_matrix)) for _ in range(len(self.adj_matrix))]
        for i in range(len(self.adj_matrix)):
            for t in range(1, len(self.adj_matrix)):
                x[i][t] = dimod.Binary(label=f'x_{i}_{t}')
        cqm.set_objective(
            sum(self.adj_matrix[0][i] * x[i][1] for i in range(1, len(self.adj_matrix))) +
            sum(self.adj_matrix[i][0] * x[i][len(self.adj_matrix) - 1] for i in range(1, len(self.adj_matrix))) +
            sum(self.adj_matrix[i][j] * x[i][t] * x[j][t + 1] for i in range(1, len(self.adj_matrix)) for j in
                range(1, len(self.adj_matrix)) if i != j for t
                in range(1, len(self.adj_matrix) - 1))
        )
        for t in range(1, len(self.adj_matrix)):
            cqm.add_constraint(sum(x[i][t] for i in range(1, len(self.adj_matrix))) == 1)
        for i in range(1, len(self.adj_matrix)):
            cqm.add_constraint(sum(x[i][t] for t in range(1, len(self.adj_matrix))) == 1)
        #utils.print_cqm_stats(cqm)
        sampler = LeapHybridCQMSampler(token=self.sapi_token)
        self.solver = 'cqm'

        solver_time_start = time()
        sample_set = sampler.sample_cqm(cqm, label="test_SoftwareX")
        sample_set.resolve()
        feasible_sampleset = sample_set.filter(lambda d: d.is_feasible)
        self.solver_time = time() - solver_time_start
        self.dwave_time = float(sample_set.info['run_time'] / 1000000)
        self.qpu_time = float(sample_set.info['qpu_access_time']) / 1000000
        self.mean_distance = float(np.mean(feasible_sampleset.data_vectors['energy']))
        self.median_distance = float(np.median(feasible_sampleset.data_vectors['energy']))
        self.unique_solutions = len(np.unique(feasible_sampleset.data_vectors['energy']))
        self.best_distance = float(np.min(feasible_sampleset.data_vectors['energy']))
        self.number_of_solutions = len(feasible_sampleset.data_vectors['energy'])
        self.best_sol_ocurrence = np.count_nonzero(feasible_sampleset.data_vectors['energy'] == self.best_distance)

        # Compute distance
        try:
            best_feasible = feasible_sampleset.first.sample

            self.best_solution = utils.decode_solution_cqm(best_feasible)
            self.best_distance = float(utils.tsp_value(self.best_solution, self.adj_matrix))
            self.best_energy = - self.best_distance
            self.mean_energy = - self.mean_distance
            self.median_energy = - self.median_distance
        except Exception as e:
            print("Error:", e)
            raise

        self.runtime = time() - start_time


########################################################################################################################
# NL SOLVER
########################################################################################################################

    def solve_with_nl(self):
        start_time = time()

        # Initialize problem data and sampler
        tsp_nl = traveling_salesperson(self.adj_matrix)
        dwave_url = 'https://cloud.dwavesys.com/sapi'
        sampler = LeapHybridNLSampler(token=self.sapi_token, endpoint=dwave_url)
        self.solver = 'nl'

        # Execute solver and extract metrics
        time_in_solver_start = time()
        futures = sampler.sample(tsp_nl, label="test_SoftwareX")
        self.solver_time = time() - time_in_solver_start
        self.dwave_time = float(futures.result().info['timing']['run_time'] / 1000000)
        self.qpu_time = float(futures.result().info['timing']['qpu_access_time']) / 1000000
        energies = []
        solution_list = []

        for i in range(tsp_nl.states.size()):
            sol = next(tsp_nl.iter_decisions()).state(i).astype(int)
            solver_objective = tsp_nl.objective.state(i)
            energies.append(solver_objective)
            solution_list.append(sol)
        self.best_distance = float(min(energies))
        self.mean_distance = float(np.mean(energies))
        self.median_distance = float(np.median(energies))
        self.unique_solutions = len(np.unique(energies))
        self.number_of_solutions = len(energies)
        self.best_solution = solution_list[energies.index(self.best_distance)]
        self.best_sol_ocurrence = energies.count(self.best_distance)
        self.best_energy = - self.best_distance
        self.mean_energy = - self.mean_distance
        self.median_energy = - self.median_distance
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

        # Create dataframe with instance analysis
        filename = 'travelling_salesman.xlsx'
        subfolder = '../outputs/tsp'
        filepath = os.path.join(subfolder, filename)
        if os.path.exists(filepath):
            df = pd.read_excel(filepath)
        else:
            df = pd.DataFrame(columns=['Instance', 'Solver', 'Instance size', 'Best distance',
                                       'Mean distance', 'Median distance',
                                       'Best solution', 'Number of solutions', 'Unique distances',
                                       'Best distances ocurrences', 'Execution time', 'DWave time', 'Time in solver',
                                       'QPU access time', 'Best energy', 'Mean energy', 'Median energy'])

        new_row = {'Instance': Path(self.instance).name, 'Solver': self.solver, 'Instance size': len(self.best_solution),
                   'Best distance': self.best_distance,
                   'Mean distance': self.mean_distance, 'Median distance': self.median_distance,
                   'Best solution': self.best_solution,
                   'Number of solutions': self.number_of_solutions, 'Unique distances': self.unique_solutions,
                   'Best distances ocurrences': self.best_sol_ocurrence, 'Execution time': self.runtime,
                   'DWave time': self.dwave_time,
                   'Time in solver': self.solver_time, 'QPU access time': self.qpu_time,
                   'Best energy': self.best_energy,
                   'Mean energy': self.mean_energy, 'Median energy': self.median_energy}

        new_df = pd.DataFrame([new_row])
        df = pd.concat([df, new_df], ignore_index=True)
        df.to_excel(filepath, index=False)
        print('Dataframe for instance', self.instance, 'has been generated')
        return df
