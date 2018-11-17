#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 20:31:57 2018

@author: Tran

PURPOSE: Computationally solve radiative transfer problems for different scenarios of optical depth
Functions: getCross_Section(s, n, *args)-
                s = depth of cloud in cm
                n = number density in cm^(-3)
                args = optical depths
                output = an array of #args x 4. The 4 cols are tau(optical depths), mean free path, N, sigma_v
            getIntensity(s, sigma_v, initI, S_v, ndensity)-
                s, ndensity = same as above
                sigma_v = cross section
                initI = initial specific Intensity at s = 0
                S_v = source function
                output = specific intensity as a function of s and cross section
            cross_sectionfn(sigma0, order, form='gaussian')-
                sigma0 = peak cross section
                order = the order of magnitude of the cross section
                    e.g. sigma0 = 3e-24, then order = -24
                    Use this order of magnitude of sigma0 to get an effective freq range
                form = linear, guassian, or constant
             return = cross-section as fn of frequency of the specified form.
Update 11/16/18:  Added plt.show() at the end and deleted plt.hold(True).
"""

import numpy as np
import matplotlib.pyplot as plt

# Note: the follow radiative transfer code assumes no scattering in the process
## Define parameters
# depth of a cloud
D = 100 #pc
pc2cm = 3.086*10**18 #cm
S = D*pc2cm #in cm
ds = 30 #step size
# number density
n = 1 #cm^(-3)
# source function
Sv = 1 #same unit as I
# initial specific intensity at s = 0
I0 = 2 #erg/s/cm^2/Hz/Sr(maybe)

### Q1: column density & optical depth
def getCross_Section(s, n, *args): 
    #args are the optical depths, s= depth of the cloud in cm, n = number density in cm^(-3)
    optdepth = args
    data = np.empty([len(optdepth), 4])
    for i, tau in enumerate(optdepth):
#         print('i, tau = ', i, tau)
        lmfp = s/tau #mean free path in cm
        N = n*lmfp  #column density in cm^(-2)
        sigmav = 1/N #cross section
        print('For an optical depth of ', tau)
        print('The column density in cm^(-2): ', N)
        print('The cross section is ', sigmav)
        data[i, :] = np.array([[tau, lmfp, N, sigmav]])
    return data
d = getCross_Section(S, n, 10**(-3), 1, 10**3) #col: tau,lmfp,N,sigmav


### Q2: Find specific intensity at any s
dist = np.linspace(0, S, ds) #cm
def getIntensity(s, sigma_v, initI, S_v, ndensity):
    # s = 1D array of the distance in cm, sigma_v = cross section, 
    # initI = initial intensity at s = 0, S_v = source function, ndensity = number density
    I_v = initI*np.e**(-s*ndensity*sigma_v) + S_v*(1-np.e**(-s*ndensity*sigma_v))
    return I_v
Ivs = getIntensity(dist,d[1, 3], I0, Sv, n)
print('Q2: Specific intensity at any s ')
print('For a cross section of ', d[1, 3])
print('Specific intensity at s = 0 is: ', Ivs[0], 'and at s = D is ', Ivs[len(dist)-1])
plt.figure(0)
plt.plot(dist, Ivs, 'b')
plt.xlabel('s (cm)')
plt.ylabel(r'$I_\nu (s)$')

### Q3: Cross-section as a function of freq
def cross_sectionfn(sigma0, order, form= 'gaussian'):
    #sigma0 = peak cross section, order= the order of magnitude of the cross section
    # e.g. sigma0 = 3e-24, then order = -24
    # Use the order of magnitude of sigma0 to get an effective freq range
    # form = linear, guassian, or constant
    fmax = 10**(-order) #max freq in Hz
    fmin = 10**(-order - 1)
    df = 100
    freq = np.linspace(fmin, fmax, df)
    if form == 'linear':
        sigma = sigma0*freq
    elif form == 'constant':
        sigma = sigma0*np.ones(len(freq))
    else: sigma = sigma0*np.exp(-np.pi*sigma0**2*(freq - fmax/2.)**2)
    return freq, sigma
y1 = cross_sectionfn(d[0,3], order=-24)
y2 = cross_sectionfn(d[1,3], order=-21)
y3 = cross_sectionfn(d[2,3], order=-18)

label1 = r'$\sigma_{v_0} = $'+ np.str(d[0,3])
label2 = r'$\sigma_{v_0} = $'+ np.str(d[1,3])
label3 = r'$\sigma_{v_0} = $'+ np.str(d[2,3])

plt.figure(1)
plt.plot(y1[0], y1[1], label=label1)
plt.xlabel('Frequency (Hz)')
plt.ylabel(r'$\sigma_v$')
plt.legend()
plt.figure(2)
plt.plot(y2[0], y2[1],label=label2)
plt.xlabel('Frequency (Hz)')
plt.ylabel(r'$\sigma_v$')
plt.legend()
plt.figure(3)
plt.plot(y3[0], y3[1],label=label3)
plt.xlabel('Frequency (Hz)')
plt.ylabel(r'$\sigma_v$')
plt.legend()


## Q4: Putting everything together
# (a) Optical depth at all frequencies tau(D) >>1
freq_a, sigconst = cross_sectionfn(3*10**(-4), order=-4, form = 'constant')  #3e-4 is sigma_v0
Inten_a = getIntensity(S, sigconst, I0, Sv, n)
plt.figure(4)
plt.title(r'$\tau_v(D) >> 1$')
plt.plot(freq_a, Inten_a, label = r'$S_v$')
plt.xlabel(r'$\nu$(Hz)')
plt.ylabel(r'$I_{\nu} (erg/s/cm^2/Hz/Sr)$')
plt.legend()
# (b) I_v(0) = 0 and tau(D) < 1
freq_b, sig_b = cross_sectionfn(6*10**(-22), order = -22)
Inten_b = getIntensity(S, sig_b, 0, Sv, n)
Sv_b = Sv*np.ones(len(freq_b))
plt.figure(5)
plt.title(r'$I_v(0) = 0, \tau_v(D) < 1$')
plt.plot(freq_b, Inten_b, 'b')
plt.plot(freq_b, Sv_b, 'r--',label = r'$S_v$')
plt.xlabel(r'$\nu$(Hz)')
plt.ylabel(r'$I_{\nu} (erg/s/cm^2/Hz/Sr)$')
plt.legend()
# (c) I_v(0) < S_v and tau(D) < 1
freq_c, sig_c = cross_sectionfn(3*10**(-21), order = -21)
Iv0_c = 1
Sv_c = 2
Iv0_c_array = Iv0_c*np.ones(len(freq_c))
Sv_c_array = Sv_c*np.ones(len(freq_c))
Inten_c = getIntensity(S, sig_c, Iv0_c, Sv_c, n)
plt.figure(6)
plt.title(r'$I_v(0) < S_v, \tau_v(D) < 1$')
plt.plot(freq_c, Inten_c, 'b')
plt.plot(freq_c, Iv0_c_array,'c--' ,label = r'$I_v(0)$')
plt.plot(freq_c, Sv_c_array, 'r--',label = r'$S_v$')
plt.xlabel(r'$\nu$(Hz)')
plt.ylabel(r'$I_{\nu} (erg/s/cm^2/Hz/Sr)$')
plt.legend()
# (d) I_v(0) > S_v and tau(D) < 1
freq_d, sig_d = cross_sectionfn(3*10**(-21), order = -21)
Iv0_d = 2
Sv_d = 1
Iv0_d_array = Iv0_d*np.ones(len(freq_d))
Sv_d_array = Sv_d*np.ones(len(freq_d))
Inten_d = getIntensity(S, sig_d, Iv0_d, Sv_d, n)
plt.figure(7)
plt.title(r'$I_v(0) > S_v, \tau_v(D) < 1$')
plt.plot(freq_d, Inten_d, 'b')
plt.plot(freq_d, Iv0_d_array,'c--' ,label = r'$I_v(0)$')
plt.plot(freq_d, Sv_d_array, 'r--',label = r'$S_v$')
plt.xlabel(r'$\nu$(Hz)')
plt.ylabel(r'$I_{\nu} (erg/s/cm^2/Hz/Sr)$')
plt.legend()
# (e) I_v(0) < S_v, tau(D) < 1, tau_v0(D) >1
freq_e, sig_e = cross_sectionfn(9*10**(-19), order = -19)
Iv0_e = 1
Sv_e = 2
Iv0_e_array = Iv0_e*np.ones(len(freq_e))
Sv_e_array = Sv_e*np.ones(len(freq_e))
Inten_e = getIntensity(S, sig_e, Iv0_e, Sv_e, n)
plt.figure(8)
plt.title(r'$I_v(0) < S_v, \tau_v(D) < 1, \tau_{v_0}(D) >1$ ')
plt.plot(freq_e, Inten_e, 'b')
plt.plot(freq_e, Iv0_e_array,'c--' ,label = r'$I_v(0)$')
plt.plot(freq_e, Sv_e_array, 'r--',label = r'$S_v$')
plt.xlabel(r'$\nu$(Hz)')
plt.ylabel(r'$I_{\nu} (erg/s/cm^2/Hz/Sr)$')
plt.legend()
# (f) I_v(0) > S_v, tau(D) < 1, tau_v0(D) >1
freq_f, sig_f = cross_sectionfn(9*10**(-19), order = -19)
Iv0_f = 2
Sv_f = 1
Iv0_f_array = Iv0_f*np.ones(len(freq_f))
Sv_f_array = Sv_f*np.ones(len(freq_f))
Inten_f = getIntensity(S, sig_f, Iv0_f, Sv_f, n)
plt.figure(9)
plt.title(r'$I_v(0) > S_v, \tau_v(D) < 1, \tau_{v_0}(D) >1$ ')
plt.plot(freq_f, Inten_f, 'b')
plt.plot(freq_f, Iv0_f_array,'c--' ,label = r'$I_v(0)$')
plt.plot(freq_f, Sv_f_array, 'r--',label = r'$S_v$')
plt.xlabel(r'$\nu$(Hz)')
plt.ylabel(r'$I_{\nu} (erg/s/cm^2/Hz/Sr)$')
plt.legend()
plt.show()
