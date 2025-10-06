import os
import numpy as np
import pandas as pd
from time import time
import networkx as nx
from pathlib import Path

import dimod
from dwave.optimization.model import Model
from dwave.system.composites import EmbeddingComposite
from dwave.system.samplers import DWaveSampler
from dwave.system import LeapHybridBQMSampler, LeapHybridCQMSampler, LeapHybridNLSampler
from qiskit_optimization.applications import Maxcut
from qiskit_optimization.converters import QuadraticProgramToQubo

import utils



class MaxCutSolver:
    def __init__(self, instance, sapi_token):

        self.instance = instance
        self.sapi_token = sapi_token
        self.solver = None
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
        self.graph = utils.create_graph_from_file(self.instance)
        self.nodes = self.graph.nodes()
        self.edges = self.graph.edges()
        self.weight_matrix = nx.to_numpy_array(self.graph)

    ########################################################################################################################
    # QPU SOLVER
    ########################################################################################################################

    def solve_with_qpu(self):
        start_time = time()

        # Initialize problem data and sampler
        max_cut = Maxcut(Maxcut.parse_gset_format(self.instance))
        qp = max_cut.to_quadratic_program()
        qp2qubo = QuadraticProgramToQubo()
        qubo = qp2qubo.convert(qp)
        bqm = dimod.as_bqm(qubo.objective.linear.to_array(), qubo.objective.quadratic.to_array(),
                                dimod.BINARY)
        self.solver = 'qpu'
        qubo = bqm.to_qubo()
        q_dict = qubo[0]
        dwave_url = 'https://cloud.dwavesys.com/sapi'
        sampler = EmbeddingComposite(DWaveSampler(token=self.sapi_token, endpoint=dwave_url))
        solver_time_start = time()
        sample_set = sampler.sample_qubo(q_dict,
                                         chain_strength=8,
                                         num_reads=10,
                                         label='MaxCut_QPU')

        self.solver_time = time() - solver_time_start
        self.dwave_time = sample_set.info['timing']['qpu_access_time']  / 1000000
        self.qpu_time = float(sample_set.info['timing']['qpu_sampling_time'] ) / 1000000
        self.number_of_solutions = len(sample_set.data_vectors['energy'])
        self.unique_solutions = len(np.unique(sample_set.data_vectors['energy']))
        self.mean_energy = np.mean(sample_set.data_vectors['energy'])
        self.median_energy = np.median(np.sort(sample_set.data_vectors['energy']))
        self.best_energy = sample_set.first.energy
        best_sol_dic = sample_set.first.sample
        self.best_solution = [best_sol_dic[i] for i in range(len(best_sol_dic))]
        self.best_sol_ocurrence = np.count_nonzero(sample_set.data_vectors['energy'] == self.best_energy)
        self.runtime = time() - start_time
        
    ########################################################################################################################
    # BQM SOLVER
    ########################################################################################################################

    def solve_with_bqm(self):
        start_time = time()

        # Initialize problem data and sampler
        max_cut = Maxcut(Maxcut.parse_gset_format(self.instance))
        qp = max_cut.to_quadratic_program()
        qp2qubo = QuadraticProgramToQubo()
        qubo = qp2qubo.convert(qp)
        bqm = (dimod.as_bqm(qubo.objective.linear.to_array(), qubo.objective.quadratic.to_array(),
                           dimod.BINARY))
        self.solver = 'bqm'
        dwave_url = 'https://cloud.dwavesys.com/sapi'
        sampler = LeapHybridBQMSampler(token=self.sapi_token, endpoint=dwave_url)

        # Execute solver and extract metrics
        solver_start_time = time()
        sample_set = sampler.sample(bqm, label="MaxCut_BQM")
        self.solver_time = time() - solver_start_time
        self.dwave_time = sample_set.info.get('run_time') / 1000000
        self.qpu_time = sample_set.info.get('qpu_access_time') / 1000000
        self.number_of_solutions = len(sample_set.data_vectors['energy'])
        self.unique_solutions = len(np.unique(sample_set.data_vectors['energy']))
        self.mean_energy = np.mean(sample_set.data_vectors['energy'])
        self.median_energy = np.median(np.sort(sample_set.data_vectors['energy']))
        self.best_energy = sample_set.first.energy
        best_sol_dic = sample_set.first.sample
        self.best_solution = [best_sol_dic[i] for i in range(len(best_sol_dic))]
        self.best_sol_ocurrence = np.count_nonzero(sample_set.data_vectors['energy'] == self.best_energy)
        self.runtime = time() - start_time

    ########################################################################################################################
    # CQM SOLVER
    ########################################################################################################################

    def solve_with_cqm(self):
        start_time = time()

        # Initialize problem data and sampler
        cqm = dimod.ConstrainedQuadraticModel()
        obj = dimod.BinaryQuadraticModel(vartype='BINARY')
        x = {i: dimod.Binary('x_{}'.format(i)) for i in range(self.graph.number_of_nodes())}
        for (i, j) in self.edges:
            obj += (- (x[int(i)] * (1 - x[int(j)]) + (1 - x[int(i)]) * x[int(j)])
                         *self.weight_matrix[int(i), int(j)])
        cqm.set_objective(obj)
        self.solver = 'cqm'
        dwave_url = 'https://cloud.dwavesys.com/sapi'
        sampler = LeapHybridCQMSampler(token=self.sapi_token, endpoint=dwave_url)

        # Execute solver and extract metrics
        solver_time_start= time()
        sample_set = sampler.sample_cqm(cqm, label="MaxCut_CQM")
        feasible_sampleset = sample_set.filter(lambda d: d.is_feasible)
        self.solver_time = time() - solver_time_start
        self.dwave_time = float(sample_set.info['run_time'] / 1000000)
        self.qpu_time = float(sample_set.info['qpu_access_time']) / 1000000
        best_sol_dic = sample_set.first.sample
        self.best_solution = utils.convert_to_bit_list(best_sol_dic)
        self.mean_energy = float(np.mean(feasible_sampleset.data_vectors['energy']))
        self.median_energy = float(np.median(feasible_sampleset.data_vectors['energy']))
        self.unique_solutions = len(np.unique(feasible_sampleset.data_vectors['energy']))
        self.best_energy = float(np.min(feasible_sampleset.data_vectors['energy']))
        self.number_of_solutions = len(feasible_sampleset.data_vectors['energy'])
        self.best_sol_ocurrence = np.count_nonzero(feasible_sampleset.data_vectors['energy'] == self.best_energy)
        self.runtime = time() - start_time

    ########################################################################################################################
    # NL SOLVER
    ########################################################################################################################

    def solve_with_nl(self):
            start_time = time()

            # Initialize problem data and sampler
            model = Model()
            weights = model.constant(utils.create_adjacency_matrix(self.graph))
            group = model.binary(self.graph.number_of_nodes())
            one = model.constant(1)
            for (i, j) in self.edges:
                if cut is None:
                    cut = (group[i] * (one - group[j]) + (one - group[i])
                                * group[j]) * weights[i][j]
                else:
                    cut = cut + (group[i] * (one - group[j]) + (one - group[i])
                                           * group[j]) * weights[i][j]
            model.minimize(- cut)
            model.lock()
            self.solver = 'nl'
            dwave_url = 'https://cloud.dwavesys.com/sapi'
            sampler = LeapHybridNLSampler(token=self.sapi_token, endpoint=dwave_url)

            # Execute solver and extract metrics
            solver_time_start= time()
            futures = sampler.sample(model, label="MaxCut_NL")
            self.solver_time = time() - solver_time_start
            self.qpu_time = futures.result().timing['qpu_access_time'] / 1000000
            self.dwave_time = futures.result().timing['runtime'] / 1000000
            self.best_energy = None
            energies = []
            solution_list = []

            for i in range(model.states.size()):
                sol = next(model.iter_decisions()).state(i).astype(int)
                solution_list.append(sol)
                solver_objective = model.objective.state(i)
                energies.append(solver_objective)
                if self.best_energy == None:
                    self.best_energy = int(solver_objective)
            self.best_energy = float(min(energies))
            self.mean_energy = float(np.mean(energies))
            self.median_energy = float(np.median(energies))
            self.unique_solutions = len(np.unique(energies))
            self.number_of_solutions = len(energies)
            self.best_solution = solution_list[energies.index(self.best_energy)]
            self.best_sol_ocurrence = energies.count(self.best_energy)
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
        filename = 'maxcut.xlsx'
        subfolder = '../outputs/mc'
        filepath = os.path.join(subfolder, filename)
        if os.path.exists(filepath):
            df = pd.read_excel(filepath)
        else:
            df = pd.DataFrame(columns=['Instance', 'Solver', 'Instance size', 'Best energy',
                                       'Mean energy', 'Median energy',
                                       'Best solution', 'Number of solutions', 'Unique energies',
                                       'Best energies ocurrences', 'Execution time', 'DWave time',
                                       'Time in solver',
                                       'QPU access time'])

        new_row = {'Instance': Path(self.instance).name, 'Solver': self.solver, 'Instance size': len(self.best_solution),
                   'Best energy': self.best_energy,
                   'Mean energy': self.mean_energy, 'Median energy': self.median_energy,
                   'Best solution': self.best_solution,
                   'Number of solutions': self.number_of_solutions, 'Unique energies': self.unique_solutions,
                   'Best energies ocurrences': self.best_sol_ocurrence, 'Execution time': self.runtime,
                   'DWave time': self.dwave_time,
                   'Time in solver': self.solver_time, 'QPU access time': self.qpu_time}

        new_df = pd.DataFrame([new_row])
        df = pd.concat([df, new_df], ignore_index=True)
        df.to_excel(filepath, index=False)
        print('Dataframe for instance', self.instance, 'has been generated')
        return df