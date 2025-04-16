import matplotlib.pyplot as plt
import numpy as np

lb_err = np.load("blocks/blocking_errors_blocking_lb.npy")
ub_err = np.load("blocks/blocking_errors_blocking_ub.npy")
lb_est = np.load("blocks/blocking_estimation_blocking_lb.npy") 
ub_est = np.load("blocks/blocking_estimation_blocking_ub.npy") 

plt.figure(figsize=(5, 5 * 6/8))
plt.errorbar(np.arange(len(ub_est)) + 1, ub_est, ub_err, color="#e8a348")
plt.xlabel("Number of blocks", fontsize=15)
plt.ylabel("Bound estimation", fontsize=15)
plt.savefig("block_ub.pdf", bbox_inches="tight")

plt.figure(figsize=(5, 5 * 6/8))
plt.errorbar(np.arange(len(lb_est)) + 1, lb_est, lb_err, color="#e8a348")
plt.xlabel("Number of blocks", fontsize=15)
plt.ylabel("Bound estimation", fontsize=15)
plt.savefig("block_lb.pdf", bbox_inches="tight")
