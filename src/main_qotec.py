import argparse
import ast
from pathlib import Path

from knapsack_solvers import KnapsackSolver
from mc_solvers import MaxCutSolver
from tsp_solvers import TravellingSalesmanSolver
from plotting import RawDataPlot, ApproximateRatioPlot

########################################################################################################################
# THIS MAIN IS A PLATFORM OF EXPERIMENTATION FOR THE MAX CUT, KNAPSACK AND TRAVELLING SALESMAN PROBLEMS WITH QPU, BQM,
# CQM AND NL SOLVERS
########################################################################################################################

if __name__ == "__main__":

    # Initialization
    parser = argparse.ArgumentParser(
        description='TS',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-s', '--solvers',
                        nargs='+',
                        help='specify tool(s) to solve the problem;')
    parser.add_argument('-i', '--instances',
                        nargs='+',
                        help='instance(s) of the problem;')
    parser.add_argument('-r', '--repetitions', type=int)
    parser.add_argument('-p', '--plots', action='store_true',
                        default=True, help='show plots (default: True)')
    parser.add_argument('-o', '--optimums',
                        type=lambda x: ast.literal_eval(x), default=None,  # Parses "[1, 3, 5]" into list
                        help="list of known optimum values e.g., [1,3,5]")


    args = parser.parse_args()
    solvers = args.solvers
    instances = args.instances
    repetitions = args.repetitions
    sapi_token = 'DGKE-e6f9971801b06c1dfee947888fda7e8b1cd28660'
    solver_map = {
        '.mc': MaxCutSolver,
        '.tsp': TravellingSalesmanSolver,
        '.knapsack': KnapsackSolver,
    }

    # Execution
    for instance in instances:
        try:
            ext = Path(instance).suffix
            problem_class = solver_map.get(ext)
            if ext =='.mc':
                problem_class = MaxCutSolver(instance=instance, sapi_token=sapi_token)
            elif ext == '.tsp':
                problem_class = TravellingSalesmanSolver(instance=instance, sapi_token=sapi_token)
            elif ext == '.kp':
                problem_class = KnapsackSolver(instance=instance, sapi_token=sapi_token)
            else:
                raise ValueError(f"Unsupported problem type: {ext}")
        except Exception as e:
            print(f"[Error] Failed to initialize problem class for '{ext}': {e}")
            continue

        for repetition in range(repetitions):
            for solver in solvers:
                problem_class.solve(solver=solver)
                dataframe = problem_class.get_dataframe()

    if args.plots:
        plotter = RawDataPlot(dataframe=dataframe)
        plotter.plot_quality_metric(solver_names=solvers, column_name='Best distance')
    if args.optimums is not None:
        plotter = ApproximateRatioPlot(dataframe=dataframe)
        plotter.plot_solver_ratio(solver_names=solvers, optimums=args.optimums, column_name='Best distance')
