import os
import pickle

from exp_params import N_ONE, DIM, MAT_MEAN, MAT_STD, NUM_ITER_SOLUTION, MAX_TIME_SOLUTION, \
    TOLERANCE_SOLUTION, EPS
from experiment_utils import ring_adj_mat, metropolis_weights, run_experiment


def exp_ring(num_nodes: int, noise: float):
    print("-------- Ring graph, {} nodes, noise = {} --------".format(num_nodes, noise))
    extragrad, sliding, z_true = run_experiment(
        n_one=N_ONE,
        d=DIM,
        mat_mean=MAT_MEAN,
        mat_std=MAT_STD,
        noise=noise,
        num_nodes=num_nodes,
        mix_mat=metropolis_weights(ring_adj_mat(num_nodes)),
        regcoef_x=2.,
        regcoef_y=2.,
        r_x=5.,
        r_y=5.,
        eps=EPS,
        num_iter_solution=NUM_ITER_SOLUTION,
        max_time_solution=MAX_TIME_SOLUTION,
        tolerance_solution=TOLERANCE_SOLUTION,
        comm_budget_experiment=200000
    )

    folder = "./logs/ring_nodes={}_noise={:.2e}".format(num_nodes, noise)
    os.system(f"mkdir -p {folder}")
    with open("/".join((folder, "extragrad_th.pkl")), "wb") as f:
        pickle.dump(extragrad.logger, f)
    with open("/".join((folder, "sliding_th.pkl")), "wb") as f:
        pickle.dump(sliding.logger, f)
    with open("/".join((folder, "z_true")), "wb") as f:
        pickle.dump(z_true, f)


if __name__ == "__main__":
    exp_ring(25, 0.0001)
    exp_ring(25, 0.001)
    exp_ring(25, 0.01)
    exp_ring(25, 0.1)
    exp_ring(25, 1.)
    exp_ring(25, 10.)
