# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 09:07:09 2024

@author: richte23
"""

from matplotlib import pyplot as plt
import numpy as np
from matplotlib.ticker import LogLocator

# Use LaTeX and serif font
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
# Global defaults: figure size, font sizes, line widths
plt.rcParams.update({
    'figure.figsize':     (5, 5*6/8),  # 5" x 3.75"
    'axes.labelsize':     15,            # axis labels
    'xtick.labelsize':    15,            # tick labels
    'ytick.labelsize':    15,
    'legend.fontsize':    12,            # legend text
    'axes.linewidth':     1,
})

# Load data
X_Data = np.loadtxt('Ca+_X_Basis10spdf_final.txt')
m_X = X_Data[:,0]
XS = X_Data[:,1]
XD3 = X_Data[:,2]
XD5 = X_Data[:,3]
XP = X_Data[:,4]

XSD5 = XD5 - XS
XSD3 = XD3 - XS
XSP  = XP  - XS
XDfine = XD5 - XD3

# F coefficients
F1P   = -1.19e-6
F1D5  = -1.557e-6
F1D3  = -1.559e-6
FDfine = F1D5 - F1D3

# Plot data
plot1Data = np.loadtxt('CaMethodsPlot1Data.txt')
yeynD5P    = plot1Data[:,0]
yeynD5Dfine= plot1Data[:,1]
yeynD5D3   = plot1Data[:,2]
yeynD5P20  = plot1Data[:,3]

plot2Data = np.loadtxt('CaMethodsPlot2Data.txt')
yeynD5DfineNL      = plot2Data[:,0]
yeynD5DfineNLsigmaX= plot2Data[:,1]
yeyenDfine124expuncertainty= plot2Data[:,2]

alpha = 1/137

# ------------------ First single plot ------------------
fig, ax = plt.subplots(dpi=100)
ax.set_xlim(1, 1e8)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_ylabel(r'$|1/(F_1 X_2-F_2 X_1)|$ ($(\mathrm{eV}^{-2} \mathrm{fm}^{2})$)')
ax.set_xlabel(r'$m_\phi$ (eV/$c^2$)')

ax.plot(m_X, np.abs(1/(F1D5*XDfine - FDfine*XSD5))/(4*np.pi*alpha),
        ls='-',  c='#e3733b', label=r'$\nu_2=\nu_{\mathrm{DD}}$', linewidth=2)
ax.plot(m_X, np.abs(1/(F1D5*XSD3 - F1D3*XSD5))/(4*np.pi*alpha),
        ls='--', c='#048c7f', label=r'$\nu_2=\nu_{732}$',       linewidth=2)
ax.plot(m_X, np.abs(1/(F1D5*XSP  - F1P*XSD5))/(4*np.pi*alpha),
        ls=':',  c='black', label=r'$\nu_2=\nu_{397}$ ($20$ Hz)',       linewidth=2)
ax.legend(loc='upper left', framealpha=1)

yticks = np.array([1e5,1e6,1e7,1e8,1e9,1e10,1e11,1e12,1e13,1e14,1e15])
ax.set_yticks(yticks)
ax.set_yticklabels(['','$10^{6}$','','$10^{8}$','','$10^{10}$','','$10^{12}$','','$10^{14}$',''])

xticks = np.array([1,1e1,1e2,1e3,1e4,1e5,1e6,1e7,1e8])
ax.set_xticks(xticks)
ax.set_xticklabels(['$1$','','$10^{2}$','','$10^{4}$','','$10^{6}$','','$10^{8}$'])

# Disable minor ticks entirely
ax.minorticks_off()

# Major tick styling only
ax.tick_params(axis='both', which='major', width=1)

plt.tight_layout()
fig.savefig('ComparisonKPlinear_plot1.pdf')

# ------------------ Second single plot ------------------
fig, ax = plt.subplots(dpi=100)
ax.set_xlim(1, 1e8)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_ylabel(r'$|\alpha_{\mathrm{NP}}/\alpha_{\mathrm{EM}}|$')
ax.set_xlabel(r'$m_\phi$ (eV/$c^2$)')

ax.plot(m_X, yeynD5Dfine/(4*np.pi*alpha),      ls='-',  c='#e3733b', linewidth=2)
ax.plot(m_X, yeynD5D3   /(4*np.pi*alpha),       ls='--', c='#048c7f',     linewidth=2)
ax.plot(m_X, yeynD5P20 /(4*np.pi*alpha),        ls=':',  c='black', linewidth=2)
ax.plot(m_X, yeynD5P   /(4*np.pi*alpha),        ls='-.', c='black', label=r'$\nu_2=\nu_{397}$ (80 kHz)', linewidth=2)
ax.legend(loc='upper left', framealpha=1)

ax.set_ylim(3.01e-12, 9.9e-3)

yticks = np.array([1e-11,1e-10,1e-9,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3])
ax.set_yticks(yticks)
ax.set_yticklabels(['$10^{-11}$','','$10^{-9}$','','$10^{-7}$','','$10^{-5}$','','$10^{-3}$'])

xticks = np.array([1,1e1,1e2,1e3,1e4,1e5,1e6,1e7,1e8])
ax.set_xticks(xticks)
ax.set_xticklabels(['$1$','','$10^{2}$','','$10^{4}$','','$10^{6}$','','$10^{8}$'])

# Disable minor ticks entirely for primary and twin axes
ax.minorticks_off()
# twin axis for y_e y_n
ax2 = ax.twinx()
yticks2 = np.array([1e-12,1e-11,1e-10,1e-9,1e-8,1e-7,1e-6,1e-5,1e-4])/(4*np.pi*alpha)
ax2.set_yscale('log')
ax2.set_ylim(ax.get_ylim())
ax2.set_yticks(yticks2)
ax2.set_yticklabels(['$10^{-12}$','','$10^{-10}$','','$10^{-8}$','','$10^{-6}$','','$10^{-4}$'])
ax2.set_ylabel(r'$|y_e y_n|/\hbar c$')
ax2.minorticks_off()

# Major tick styling only
ax.tick_params(axis='both', which='major', width=1)
ax2.tick_params(axis='both', which='major', width=1)

plt.tight_layout()
fig.savefig('ComparisonKPlinear_plot2.pdf')





#####################################################################################################################





fig, axs = plt.subplots(dpi=100)  

# First plot
axs.set_xscale('log')
axs.set_yscale('log')
axs.set_ylabel('$|\\alpha_{\mathrm{NP}}/\\alpha_{\mathrm{EM}}|$')
axs.set_xlabel('$m_\phi$ (eV/$c^2$)')
#axs.plot(m_X, yeynD5Dfine2, ls='-', c='black', label='linear theory data', linewidth=3)
axs.plot(m_X, np.array(yeyenDfine124expuncertainty)/ (4 * np.pi * alpha), ls='-', c='black', label='Linear mock data', linewidth=2, alpha=0.8)
axs.plot(m_X, np.array(yeynD5DfineNL)/ (4 * np.pi * alpha), ls='--', c='#e3733b', label='Exp. data $\sigma[X]=0$ ', linewidth=2)
axs.plot(m_X, np.array(yeynD5DfineNLsigmaX)/ (4 * np.pi * alpha), ls=':', c='#048c7f', label='Exp. data $\sigma[X] > 0$', linewidth=2)
axs.legend(loc='upper left')

axs.set_xlim(1,10**8)
axs.set_ylim(1.0001*10**-10,9.999*10**-3)
# Create a twin Axes sharing the same x-axis

xticks = np.array([1, 1e1,1e2,1e3, 1e4, 1e5,1e6,1e7,1e8])
axs.set_xscale('log')

axs.set_xticks(xticks)
axs.set_xticklabels(['$1$','','$10^{2}$','','$10^{4}$','','$10^{6}$','','$10^{8}$'])

yticks = np.array([1e-9, 1e-8,1e-7,1e-6, 1e-5, 1e-4,1e-3])
axs.set_yscale('log')

axs.set_yticks(yticks)
axs.set_yticklabels(['$10^{-9}$','','$10^{-7}$','','$10^{-5}$','','$10^{-3}$'])

# Create a twin Axes sharing the same x-axis
axs2 = axs.twinx()

# Calculate and set y-axis ticks for y_e y_n / (4 * pi * alpha)
alpha = 1/137  # Fine structure constant
alpha = 1/137  # Fine structure constant
yticks = np.array([1e-10,1e-9, 1e-8,1e-7, 1e-6,1e-5, 1e-4]) / (4 * np.pi * alpha)
axs2.set_yscale('log')
axs2.set_ylim(axs.get_ylim())
axs2.set_yticks(yticks)
axs2.set_yticklabels(['$10^{-10}$','','$10^{-8}$','','$10^{-6}$','','$10^{-4}$'])
axs2.set_ylabel('$|y_e y_n|/\hbar c$')

# Adjust layout to prevent clipping of labels
plt.tight_layout()
axs.tick_params(axis='both', which='major')
axs2.tick_params(axis='both', which='major')

axs.tick_params(axis='both', which='minor')
axs2.tick_params(axis='both', which='minor')
# Save the combined figure as a PDF


min_loc = LogLocator(subs='all', numticks=10)

# fig.text(0.5, 0.952, '$\\nu_1=\\nu_{729}$, $\\nu_2=\\nu_{\mathrm{DD}}$', ha='center', va='center', fontsize=24,bbox=dict(boxstyle="round",fc=(1., 1, 1),ec=(0.2, 0.2, 0.2)))

axs.xaxis.set_minor_locator(min_loc)
#axs.yaxis.set_minor_locator(min_loc)
plt.savefig('ComparisonKPnonlinear_plot.pdf')








#####################################################################################################################################################












fig, axs = plt.subplots(dpi=100)  # Adjust figsize as needed

alpha = 1/137
# First plot
axs.set_xlim(1,10**8)
axs.set_xscale('log')
xticks = np.array([1, 1e1,1e2,1e3, 1e4, 1e5,1e6,1e7,1e8])
axs.set_xscale('log')

axs.set_xticks(xticks)
axs.set_xticklabels(['$1$','','$10^{2}$','','$10^{4}$','','$10^{6}$','','$10^{8}$'])
#axs.set_yscale('log')
axs.set_ylabel('New physics coefficient $X$ (eV)')
axs.set_xlabel('$m_\phi$ (eV/$c^2$)')
#axs.plot(m_X, yeynD5Dfine2, ls='-', c='black', label='linear theory data', linewidth=3)
axs.plot(m_X, XSD5* (4 * np.pi * alpha), ls='-', c='#e3733b', label='$\\nu_{729}$', linewidth=2)
axs.plot(m_X, XSP* (4 * np.pi * alpha), ls='--', c='#048c7f', label='$\\nu_{397}$ ', linewidth=2)
#axs.plot(m_X, yeynD5DfineNLsigmaX, ls=':', c='#048c7f', label='experimental data $\sigma[X] > 0$', linewidth=3)
axs.legend(loc='upper right')


plt.tight_layout()

min_loc = LogLocator(subs='all', numticks=10)

axs.xaxis.set_minor_locator(min_loc)
# Adjust layout to prevent clipping of labels
plt.tight_layout()
axs.tick_params(axis='both', which='major')
axs2.tick_params(axis='both', which='major')

axs.tick_params(axis='both', which='major')
axs2.tick_params(axis='both', which='major')

# Save the combined figure as a PDF
plt.savefig('X_Ca+.pdf')





#############################################################################################################################################









mPhi = np.loadtxt('Yb24_GKP_AME20_X/Yb24_GKP_AME20_X_1.dat', usecols=(0))


Yb24_GKP_AME20_X1=np.loadtxt('Yb24_GKP_AME20_X/Yb24_GKP_AME20_X_1.dat', usecols=(1))
Yb24_GKP_AME20_X2=np.loadtxt('Yb24_GKP_AME20_X/Yb24_GKP_AME20_X_2.dat', usecols=(1))
Yb24_GKP_AME20_X3=np.loadtxt('Yb24_GKP_AME20_X/Yb24_GKP_AME20_X_3.dat', usecols=(1))
Yb24_GKP_AME20_X4=np.loadtxt('Yb24_GKP_AME20_X/Yb24_GKP_AME20_X_4.dat', usecols=(1))
Yb24_GKP_AME20_X5=np.loadtxt('Yb24_GKP_AME20_X/Yb24_GKP_AME20_X_5.dat', usecols=(1))
Yb24_GKP_AME20_X6=np.loadtxt('Yb24_GKP_AME20_X/Yb24_GKP_AME20_X_6.dat', usecols=(1))
Yb24_GKP_AME20_X7=np.loadtxt('Yb24_GKP_AME20_X/Yb24_GKP_AME20_X_7.dat', usecols=(1))
Yb24_GKP_AME20_X8=np.loadtxt('Yb24_GKP_AME20_X/Yb24_GKP_AME20_X_8.dat', usecols=(1))
Yb24_GKP_AME20_X9=np.loadtxt('Yb24_GKP_AME20_X/Yb24_GKP_AME20_X_9.dat', usecols=(1))
Yb24_GKP_AME20_X10=np.loadtxt('Yb24_GKP_AME20_X/Yb24_GKP_AME20_X_10.dat', usecols=(1))
Yb24_GKP_AME20_Xenvelope=np.loadtxt('Yb24_GKP_AME20_X/Yb24_GKP_AME20_X_envelope.dat', usecols=(1))

Yb24_GKP_Door_X1=np.loadtxt('Yb24_GKP_Door_X/Yb24_GKP_Door_X_1.dat', usecols=(1))
Yb24_GKP_Door_X2=np.loadtxt('Yb24_GKP_Door_X/Yb24_GKP_Door_X_2.dat', usecols=(1))
Yb24_GKP_Door_X3=np.loadtxt('Yb24_GKP_Door_X/Yb24_GKP_Door_X_3.dat', usecols=(1))
Yb24_GKP_Door_X4=np.loadtxt('Yb24_GKP_Door_X/Yb24_GKP_Door_X_4.dat', usecols=(1))
Yb24_GKP_Door_X5=np.loadtxt('Yb24_GKP_Door_X/Yb24_GKP_Door_X_5.dat', usecols=(1))
Yb24_GKP_Door_X6=np.loadtxt('Yb24_GKP_Door_X/Yb24_GKP_Door_X_6.dat', usecols=(1))
Yb24_GKP_Door_X7=np.loadtxt('Yb24_GKP_Door_X/Yb24_GKP_Door_X_7.dat', usecols=(1))
Yb24_GKP_Door_X8=np.loadtxt('Yb24_GKP_Door_X/Yb24_GKP_Door_X_8.dat', usecols=(1))
Yb24_GKP_Door_X9=np.loadtxt('Yb24_GKP_Door_X/Yb24_GKP_Door_X_9.dat', usecols=(1))
Yb24_GKP_Door_X10=np.loadtxt('Yb24_GKP_Door_X/Yb24_GKP_Door_X_10.dat', usecols=(1))
Yb24_GKP_Door_Xenvelope=np.loadtxt('Yb24_GKP_Door_X/Yb24_GKP_Door_X_envelope.dat', usecols=(1))

Yb24_NMGKP_X1=np.loadtxt('Yb24_NMGKP_X/Yb24_NMGKP_X_1.dat', usecols=(1))
Yb24_NMGKP_X2=np.loadtxt('Yb24_NMGKP_X/Yb24_NMGKP_X_2.dat', usecols=(1))
Yb24_NMGKP_X3=np.loadtxt('Yb24_NMGKP_X/Yb24_NMGKP_X_3.dat', usecols=(1))
Yb24_NMGKP_X4=np.loadtxt('Yb24_NMGKP_X/Yb24_NMGKP_X_4.dat', usecols=(1))
Yb24_NMGKP_X5=np.loadtxt('Yb24_NMGKP_X/Yb24_NMGKP_X_5.dat', usecols=(1))
Yb24_NMGKP_Xenvelope=np.loadtxt('Yb24_NMGKP_X/Yb24_NMGKP_X_envelope.dat', usecols=(1))























fig, axs = plt.subplots(dpi=100)  # Adjust figsize as needed

# First plot
axs.set_xscale('log')
axs.set_yscale('log')
axs.set_ylabel('$|\\alpha_{\mathrm{NP}}/\\alpha_{\mathrm{EM}}|$')
axs.set_xlabel('$m_\phi$ (eV/$c^2$)')


#axs.plot(m_X, yeynD5Dfine2, ls='-', c='black', label='linear theory data', linewidth=3)
# axs.plot(mPhi, Yb24_GKP_AME20_Xenvelope, ls='-', c='green', label='3D GKP (AME2020)', linewidth=4)
# axs.plot(mPhi, Yb24_GKP_AME20_X1, ls='-',c='green', linewidth=3,alpha=0.2)
# axs.plot(mPhi, Yb24_GKP_AME20_X2, ls='-',c='green', linewidth=3,alpha=0.2)
# axs.plot(mPhi, Yb24_GKP_AME20_X3, ls='-',c='green', linewidth=3,alpha=0.2)
# axs.plot(mPhi, Yb24_GKP_AME20_X4, ls='-',c='green', linewidth=3,alpha=0.2)
# axs.plot(mPhi, Yb24_GKP_AME20_X5, ls='-',c='green', linewidth=3,alpha=0.2)
# axs.plot(mPhi, Yb24_GKP_AME20_X6, ls='-',c='green', linewidth=3,alpha=0.2)
# axs.plot(mPhi, Yb24_GKP_AME20_X7, ls='-',c='green', linewidth=3,alpha=0.2)
# axs.plot(mPhi, Yb24_GKP_AME20_X8, ls='-',c='green', linewidth=3,alpha=0.2)
# axs.plot(mPhi, Yb24_GKP_AME20_X9, ls='-',c='green', linewidth=3,alpha=0.2)
# axs.plot(mPhi, Yb24_GKP_AME20_X10, ls='-',c='green', linewidth=3,alpha=0.2)




axs.plot(mPhi, Yb24_GKP_Door_Xenvelope, ls='-', c='#048c7f', label='3D GKP (Door2024)', linewidth=2)
axs.plot(mPhi, Yb24_GKP_Door_X1, ls='-',c='#048c7f', linewidth=1,alpha=0.2)
axs.plot(mPhi, Yb24_GKP_Door_X2, ls='-',c='#048c7f', linewidth=1,alpha=0.2)
axs.plot(mPhi, Yb24_GKP_Door_X3, ls='-',c='#048c7f', linewidth=1,alpha=0.2)
axs.plot(mPhi, Yb24_GKP_Door_X4, ls='-',c='#048c7f', linewidth=1,alpha=0.2)
axs.plot(mPhi, Yb24_GKP_Door_X5, ls='-',c='#048c7f', linewidth=1,alpha=0.2)
axs.plot(mPhi, Yb24_GKP_Door_X6, ls='-',c='#048c7f', linewidth=1,alpha=0.2)
axs.plot(mPhi, Yb24_GKP_Door_X7, ls='-',c='#048c7f', linewidth=1,alpha=0.2)
axs.plot(mPhi, Yb24_GKP_Door_X8, ls='-',c='#048c7f', linewidth=1,alpha=0.2)
axs.plot(mPhi, Yb24_GKP_Door_X9, ls='-',c='#048c7f', linewidth=1,alpha=0.2)
axs.plot(mPhi, Yb24_GKP_Door_X10, ls='-',c='#048c7f',linewidth=1,alpha=0.2)




axs.plot(mPhi, Yb24_NMGKP_Xenvelope, ls='-', c='#e3733b', label='4D NMGKP ', linewidth=2)
axs.plot(mPhi, Yb24_NMGKP_X1, ls='-',c='#e3733b', linewidth=1,alpha=0.2)
axs.plot(mPhi, Yb24_NMGKP_X2, ls='-',c='#e3733b', linewidth=1,alpha=0.2)
axs.plot(mPhi, Yb24_NMGKP_X3, ls='-',c='#e3733b', linewidth=1,alpha=0.2)
axs.plot(mPhi, Yb24_NMGKP_X4, ls='-',c='#e3733b', linewidth=1,alpha=0.2)
axs.plot(mPhi, Yb24_NMGKP_X5, ls='-',c='#e3733b', linewidth=1,alpha=0.2)




axs.legend()


axs.set_xlim(1,10**8)
axs.set_ylim(2*10**-11,5*10**-3)
# Create a twin Axes sharing the same x-axis
xticks = np.array([1, 1e1,1e2,1e3, 1e4, 1e5,1e6,1e7,1e8])
axs.set_xscale('log')

axs.set_xticks(xticks)
axs.set_xticklabels(['$1$','','$10^{2}$','','$10^{4}$','','$10^{6}$','','$10^{8}$'])


yticks = np.array([1e-10,1e-9, 1e-8,1e-7, 1e-6,1e-5, 1e-4,1e-3])
axs.set_yscale('log')
axs.set_ylim(axs.get_ylim())
axs.set_yticks(yticks)
axs.set_yticklabels(['','$10^{-9}$','','$10^{-7}$','','$10^{-5}$','','$10^{-3}$'])

min_loc = LogLocator(subs='all', numticks=10)

axs.xaxis.set_minor_locator(min_loc)
#axs.yaxis.set_minor_locator(LogLocator(subs='all', numticks=1000))

axs2 = axs.twinx()

# Calculate and set y-axis ticks for y_e y_n / (4 * pi * alpha)
alpha = 1/137  # Fine structure constant
yticks = np.array([1e-11,1e-10,1e-9, 1e-8,1e-7, 1e-6,1e-5, 1e-4])/ (4 * np.pi * alpha)
axs2.set_yscale('log')
axs2.set_ylim(axs.get_ylim())
axs2.set_yticks(yticks)
axs2.set_yticklabels(['','$10^{-10}$','','$10^{-8}$','','$10^{-6}$','','$10^{-4}$'])
axs2.set_ylabel('$|y_e y_n|/\hbar c$')



# Adjust layout to prevent clipping of labels
plt.tight_layout()
axs.tick_params(axis='both', which='major')
axs2.tick_params(axis='both', which='major')

axs.tick_params(axis='both', which='minor')
axs2.tick_params(axis='both', which='minor')

# Save the combined figure as a PDF
plt.savefig('Yb_GKP_NMK_envelope.pdf')






