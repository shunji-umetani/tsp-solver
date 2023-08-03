# tsp-solver
<img width="400" alt="tour" src="https://user-images.githubusercontent.com/58223377/172106109-650e058f-1c5b-4371-b440-ce0869f07333.png">

metaheuristics for traveling salesman problem (TSP)
- `tsp_ls_naive.py` [Local Search (LS)](https://github.com/shunji-umetani/tsp-solver/blob/main/tsp_ls_naive.py)
- `tsp_ls_nblist.py` [Local Search with Neighbor-List (LS-NL)](https://github.com/shunji-umetani/tsp-solver/blob/main/tsp_ls_nblist.py)
- `tsp_mls.py` [Random Multi-start Local Search (MLS)](https://github.com/shunji-umetani/tsp-solver/blob/main/tsp_mls.py)
- `tsp_ils.py` [Iterated Local Search (ILS)](https://github.com/shunji-umetani/tsp-solver/blob/main/tsp_ils.py)
- `tsp_ils_fls.py` [Iterated Local Search with Fast Local Search (ILS-FLS)](https://github.com/shunji-umetani/tsp-solver/blob/main/tsp_ils_fls.py)
- `tsp_ils_imp.py` [Improved ILS-FLS (ILS+)](https://github.com/shunji-umetani/tsp-solver/blob/main/tsp_ils_imp.py)
- `tsp_gls.py` [Guided Local Search (GLS)](https://github.com/shunji-umetani/tsp-solver/blob/main/tsp_gls.py)
- `tsp_gls_fls.py` [Guided Local Search with FLS (GLS-FLS)](https://github.com/shunji-umetani/tsp-solver/blob/main/tsp_gls_fls.py)
- `tsp_grasp.py` [Greedy Randomized Adaptive Search Procedure (GRASP)](https://github.com/shunji-umetani/tsp-solver/blob/main/tsp_grasp.py)
- `tsp_sa.py` [Simulated Annealing (SA)](https://github.com/shunji-umetani/tsp-solver/blob/main/tsp_sa.py)
- `tsp_tabu_rule1.py` [Tabu Search with Rule1 (TS1)](https://github.com/shunji-umetani/tsp-solver/blob/main/tsp_tabu_rule1.py)
- `tsp_tabu_rule2.py` [Tabu Search with Rule2 (TS2)](https://github.com/shunji-umetani/tsp-solver/blob/main/tsp_tabu_rule2.py)
- `tsp_ma.py` [Memetic Alsogithm (MA)](https://github.com/shunji-umetani/tsp-solver/blob/main/tsp_ma.py)
- `tsp_ma_fls.py` [Memetic Alsogithm with FLS (MA-FLS)](https://github.com/shunji-umetani/tsp-solver/blob/main/tsp_ma_fls.py)

## Feature
- Simple implementation of metaheuristics in Python.
- Local search with 2-opt, Or-opt, and 3-opt neighborhood search.
- Efficient LS implementation using neighbor-list.
- For Euclidean TSP instances (EUC2D) in [TSPLIB](http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/).

## Usage
common usage for all codes
```
$ tsp_ils.py [-h] [-t TIME] [-d] filename
```
- `filename` TSP instance (mandatory)
- `-d` visually display obtained tour (optional)
- `-t` timelimit (optional, default 60 sec) except for `tsp_ls_naive.py` and `tsp_ls_nblist.py` 

## References
[slide (Japanese)](https://speakerdeck.com/umepon/introduction-to-design-and-implementation-of-metaheuristics)

## Author
[Umetani, Shunji](https://github.com/shunji-umetani)

## License
This software is released under the MIT License, see LICENSE.
