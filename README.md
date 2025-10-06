# softwarex_benchmarking



## Getting started

This is a platform for benchmarking combinatorial optimization problem instances with DWave's Hybrid Solvers: QPU, BQM, CQM and NL.
The interested user should have a DWave account and a valid API key (see https://www.dwavesys.com/quantum-launchpad/).
\
\
To execute the benchmarking, the user should run the following command: python main_qotec.py -i 'INSTANCES' -s 'SOLVERS' -r 'REPETITIONS' -p 'PLOTS' -o 'OPTIMUMS'. For example: python main_qotec.py -i ../instances/mc/MaxCut_10.mc ../instances/mc/MaxCut_20.mc ../instances/mc/MaxCut_40.mc -s bqm cqm -r 5 -p False -p "[-17, -22, -33]".