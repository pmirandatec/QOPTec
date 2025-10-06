import numpy as np
import matplotlib.pyplot as plt


class ApproximateRatioPlot:

    def __init__(self, dataframe, minimization=True):
        self.dataframe = dataframe
        self.minimization = minimization
        self.solvers = self.dataframe['Solver'].unique()
        self.fig, self.ax = plt.subplots()


    def plot_solver_ratio(self, solver_names, optimums,  column_name='Best energy',):
        import numpy as np

        # Always ensure solver_names is a list
        if isinstance(solver_names, str):
            solver_names = [solver_names]

        for solver_name in solver_names:
            # Filter the dataframe for the given solver
            energies = self.dataframe[self.dataframe['Solver'] == solver_name]

            # Skip plotting if no data exists for this solver at all
            if energies.empty:
                print(f" No data found for solver '{solver_name}'")
                continue

            # Compute mean & std grouped by instance size
            energies_mean = energies.groupby('Instance size')[column_name].mean().reset_index()
            std = energies.groupby('Instance size')[column_name].std().reset_index()

            # Get all instance sizes in the dataset
            all_instance_sizes = np.sort(self.dataframe['Instance size'].unique())

            # Reindex to include all instance sizes, filling missing ones with NaN
            energies_mean = energies_mean.set_index('Instance size').reindex(all_instance_sizes).reset_index()
            std = std.set_index('Instance size').reindex(all_instance_sizes).reset_index()

            # Match optimums to all instance sizes
            # Assuming optimums are ordered according to all_instance_sizes
            if len(optimums) != len(all_instance_sizes):
                raise ValueError(
                    f"Optimums size ({len(optimums)}) doesn't match total instance sizes ({len(all_instance_sizes)}). "
                    f"Provide optimums for all instance sizes."
                )

            # Calculate the approximation ratio safely (NaNs handled automatically)
            if self.minimization:
                approximate_ratio = optimums / energies_mean[column_name]
                std_ratio = (std[column_name] * optimums) / (energies_mean[column_name] ** 2)
            else:
                approximate_ratio = energies_mean[column_name] / optimums
                std_ratio = std[column_name] / optimums

            # Plot only where data exists (NaNs skipped automatically)
            self.ax.errorbar(
                energies_mean['Instance size'],
                y=approximate_ratio,
                yerr=std_ratio,
                fmt='-o',
                label=solver_name,
                capsize=5
            )

            # Fill area around error bars where data exists
            self.ax.fill_between(
                energies_mean['Instance size'],
                approximate_ratio - std_ratio,
                approximate_ratio + std_ratio,
                alpha=0.3
            )

        # Set labels and title
        self.ax.set_xlabel('Instance Size')
        self.ax.set_ylabel('Approximation ratio')
        self.ax.set_title('')
        plt.grid(True)

        # Add legend
        self.ax.legend()

        # Display the plot
        plt.draw()
        plt.show()

class RawDataPlot:

    def __init__(self, dataframe, minimization=True):
        self.dataframe = dataframe
        self.minimization = minimization
        self.solvers = self.dataframe['Solver'].unique()
        self.fig, self.ax = plt.subplots()

    def plot_quality_metric(self, solver_names, column_name='Best energy'):
        """
        Plots the quality metric for one or multiple solvers.
        Handles missing data automatically.
        """
        # Ensure solver_names is always a list
        if isinstance(solver_names, str):
            solver_names = [solver_names]
        elif solver_names is None:  # Automatically use all solvers if None is passed
            solver_names = self.solvers

        # Get all possible instance sizes in the dataset
        all_instance_sizes = np.sort(self.dataframe['Instance size'].unique())

        for solver_name in solver_names:
            # Filter the dataframe for the current solver
            energies = self.dataframe[self.dataframe['Solver'] == solver_name]

            # Skip if solver has no data at all
            if energies.empty:
                print(f"⚠️ Skipping solver '{solver_name}' — no data found.")
                continue

            # Calculate mean & std for the solver
            energies_mean = energies.groupby('Instance size')[column_name].mean().reset_index()
            std = energies.groupby('Instance size')[column_name].std().reset_index()

            # Align solver data to all instance sizes (fill missing with NaN)
            energies_mean = energies_mean.set_index('Instance size').reindex(all_instance_sizes).reset_index()
            std = std.set_index('Instance size').reindex(all_instance_sizes).reset_index()

            # Plot solver data (missing points are skipped automatically)
            self.ax.errorbar(
                energies_mean['Instance size'],
                y=energies_mean[column_name],
                yerr=std[column_name],
                fmt='-o',
                label=solver_name,
                capsize=5
            )

            # Fill area around error bars where data exists
            self.ax.fill_between(
                energies_mean['Instance size'],
                energies_mean[column_name] - std[column_name],
                energies_mean[column_name] + std[column_name],
                alpha=0.3
            )

        # Set labels and title
        self.ax.set_xlabel('Instance Size')
        self.ax.set_ylabel(column_name)
        self.ax.set_title('')
        plt.grid(True)

        # Add legend
        self.ax.legend()

        # Display the plot
        plt.draw()
        plt.show()

