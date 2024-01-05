from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from kifit.loadelems import Elem
from kifit.performfit import perform_odr, perform_linreg, linfit

from scipy.stats import multivariate_normal
from scipy.linalg import cholesky

np.random.seed(1)

ca = Elem.get('Ca_testdata')

num_samples = 5000

print("alphaNP_init", ca.alphaNP_init)
print("sig_alphaNP_init", ca.sig_alphaNP_init)

alphaNP_samples = np.random.normal(ca.alphaNP_init, ca.sig_alphaNP_init, num_samples)

ca_fit_params = np.tensordot(np.ones(num_samples), ca.means_fit_params, axes=0)
ca_fit_params[:, -1] = alphaNP_samples

print("fit params", ca_fit_params)

print("alphaNP")
print(alphaNP_samples)
# print(ca_fit_params[:,-1])
# print(np.average(ca_fit_params[:, -1]))
# print(np.std(ca_fit_params[:, -1]))
print(np.average(alphaNP_samples))
print(np.std(alphaNP_samples))


absd_init = ca.absd
absd_samples = []
for i in range(num_samples):
    if i % (num_samples // 100) == 0:
        prog = np.round(i / num_samples * 100, 1)
        print("Progress", prog, "%")
    # ca._update_elem_params(ca_elem_params[i])
    ca._update_fit_params(ca_fit_params[i])
    absd_samples.append(ca.absd)


absd_samples = np.array(absd_samples)
# cov_absd_np = np.cov(absd_samples, rowvar=False)

mean_absd = np.average(absd_samples, axis=0)

dev_absd_samples = np.array([[(absd_samples[s, a] - mean_absd[a])
    for a in range(ca.nisotopepairs)] for s in range(num_samples)])

simple_LL = 1 / 2 * (
    np.array([np.sum(np.array([
        absd_samples[s, a]**2 / dev_absd_samples[s, a]**2
        for a in range(ca.nisotopepairs)]))
        + np.sum(np.array([np.log(dev_absd_samples[s, b]**2)
            for b in range(ca.nisotopepairs)]))
        for s in range(num_samples)]))

# print("absd", absd_samples)
# print("dev ", dev_absd_samples)
#
# print("LL", simple_LL)

fig, ax = plt.subplots()
ax.scatter(ca_fit_params.T[-1], simple_LL)
plt.tight_layout()
plt.savefig("llplot.pdf")
ax.set_ylim(0, 1e-1)
# for a in range(ca.nisotopepairs):
#
#     fig, ax = plt.subplots()
#
#     ax.scatter(ca_fit_params.T[-1], absd_samples[:, a])
#     plt.tight_layout()
#     plt.show()
#

simplest_LL = np.sum(absd_samples**2, axis=1)
print("simplest_LL", simplest_LL.shape)

fig, ax = plt.subplots()
ax.scatter(ca_fit_params.T[-1], simplest_LL)
plt.savefig("dsqplot.pdf")

devilist= np.sum(dev_absd_samples**2, axis=1)

fig, ax = plt.subplots()
ax.scatter(ca_fit_params.T[-1], devilist, label="deviations")
plt.savefig("devilplot.pdf")

testi = np.sum(absd_samples**2, axis=1) / np.sum(dev_absd_samples**2, axis=1)

fig, ax = plt.subplots()
ax.scatter(ca_fit_params.T[-1], testi, label="testi")
ax.set_ylim(0, 20)
plt.savefig("testiplot.pdf")


cov_absd_samples = np.array([[[
    (absd_samples[s, a] - mean_absd[a]) * (absd_samples[s, b] - mean_absd[b])
    for a in range(ca.nisotopepairs)] for b in range(ca.nisotopepairs)]
    for s in range(num_samples)])


mean_absd = np.average(absd_samples, axis=0)

cov_absd = np.cov(absd_samples, rowvar=False)
print("cov_absd)")
print(cov_absd)

def choLL(absd, covmat):

    chol_covmat = cholesky(covmat, lower=True)

    A = np.linalg.solve(chol_covmat, absd)
    At = np.linalg.solve(chol_covmat.T, absd)

    ll = (np.dot(A, A) / 2 + np.dot(At, At) / 2
        + np.sum(np.log(np.diag(chol_covmat)))
        + len(absd) / 2 * np.log(2 * np.pi))

    return ll


chollist = []
for s in range(num_samples):
    chollist.append(choLL(absd_samples[s], cov_absd))

fig, ax = plt.subplots()
ax.scatter(ca_fit_params.T[-1], chollist, label="cholli")
# ax.set_xlim(-1e-8, 1e-8)
# ax.set_ylim(0, 1e3)
ax.set_xlim(-1e-11, 1e-11)
plt.savefig("cholliplot.pdf")

##############################################################################

ca_elem_params = multivariate_normal.rvs(
    ca.means_input_params,
    np.diag(ca.stdevs_input_params**2),
    size=num_samples
)
# ca_fit_params = multivariate_normal.rvs(
#     ca.means_fit_params,
#     np.diag(ca.stdevs_fit_params),
#     size=num_samples
# )

absd_samples = []
for i in range(num_samples):
    if i % (num_samples // 100) == 0:
        prog = np.round(i / num_samples * 100, 1)
        print("Progress", prog, "%")
    ca._update_elem_params(ca_elem_params[i])
    ca._update_fit_params(ca_fit_params[i])
    absd_samples.append(ca.absd)


absd_samples = np.array(absd_samples)

cov_absd = np.cov(absd_samples, rowvar=False)

print("cov_absd)")
print(cov_absd)

print("alphas")
print(ca_fit_params.T[-1])

chollist = []
for s in range(num_samples):
    chollist.append(choLL(absd_samples[s], cov_absd))

fig, ax = plt.subplots()
ax.scatter(ca_fit_params.T[-1], chollist, label="cholli")
# ax.set_xlim(-1e-8, 1e-8)
# ax.set_ylim(0, 1e3)
plt.savefig("wigglycholliplot.pdf")

###############################################################################

ca_elem_params = multivariate_normal.rvs(
    ca.means_input_params,
    np.diag(ca.stdevs_input_params**2),
    size=num_samples
)
ca_fit_params = multivariate_normal.rvs(
    ca.means_fit_params,
    np.diag(ca.stdevs_fit_params**2),
    size=num_samples
)

absd_samples = []

for i in range(num_samples):
    if i % (num_samples // 100) == 0:
        prog = np.round(i / num_samples * 100, 1)
        print("Progress", prog, "%")
    ca._update_elem_params(ca_elem_params[i])
    ca._update_fit_params(ca_fit_params[i])
    absd_samples.append(ca.absd)

absd_samples = np.array(absd_samples)
cov_absd = np.cov(absd_samples, rowvar=False)


print("alphas")
print(ca_fit_params.T[-1])

chollist = []
for s in range(num_samples):
    chollist.append(choLL(absd_samples[s], cov_absd))

fig, ax = plt.subplots()
ax.scatter(ca_fit_params.T[-1], chollist, label="cholli")
# ax.set_xlim(-1e-8, 1e-8)
# ax.set_ylim(0, 1e3)
plt.savefig("evenwiggliercholliplot.pdf")



###############################################################################

# mean_absd = np.average(absd_samples, axis=0)
#
# cov_absd_samples = np.array([[[
#     (absd_samples[s, a] - mean_absd[a]) * (absd_samples[s, b] - mean_absd[b])
#     for a in range(ca.nisotopepairs)] for b in range(ca.nisotopepairs)]
#     for s in range(num_samples)])  # not positive definite most of the time
#
#
# print("cov_absd_samples[0]")
# print(cov_absd_samples[0])
#
# print("alphas")
# print(ca_fit_params.T[-1])
#
# chollist = []
# for s in range(num_samples):
#     chollist.append(choLL(absd_samples[s], cov_absd_samples[s]))
#
# fig, ax = plt.subplots()
# ax.scatter(ca_fit_params.T[-1], chollist, label="cholli")
# ax.set_xlim(-1e-8, 1e-8)
# ax.set_ylim(0, 1e3)
