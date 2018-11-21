#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 12:55:32 2018

@author: Tran

PURPOSE: Simulate the interaction between an electron and a charged particle. 
         Calculate the acceleration and freq spectrum of radiation of e-.
Functions: euler(t0, r0, v0, dt, N, q1,q2,m,k)- uses the euler method to numerically
                solve differential equation.
                t0 = initial time
                r0, v0 = 1D array-like of initial position and velocity in both 
                x and y, assumming r0[0] is x0 and r0[1] = y0. Same for v0.
                dt = time interval to estimate over
                N = number of steps
                q1, q2 = charges(C) of the particles: electron and ion
                m,k = mass of electron(kg), Couloumb's constant
                outputs= return t (1D), r, v, a array of x and y components
            getpower(t, s, v, dt, n, q1, q2, m, k, mode = 'Power')- FFT a and 
                compute a power spectrum.
                t, s, v, dt, n, q1, q2, m, k= args for calling the euler fn
                mode = default to 'Power'-> only return the necessary components
                for plotting the power spectrum wrt to freq.
"""

import numpy as np
import matplotlib.pyplot as plt
########################## Defining parameters ################################
"""
Assumptions:
    - The ion, Ze, is at the origin. 
    - x and y axes will be in unit of Bohr radius, a0
"""
c = 3e8 #m/s, speed of light
v0x = 5*0.001*c #initial velocity of the electron in x-direction
v0y = 0
a0 = 5.29e-11 #m: Bohr radius
y0 = 300*a0 #~106Angstrom initial position of the electron, away from the ion
x0 = -300*a0
k = 8.99e9 #N·m^2·C^−2: Couloumb constant
q = 1.6e-19 #C : charge of electron
z = 1  #ion #
m = 9.1e-31 #kg, mass of electron
###############################################################################
##################### Calc position & acceleration of e-#######################
## euler method to numerically solve the differential eq
def euler(t0, r0, v0, dt, N, q1,q2,m,k):
    """
    t0 = initial time
    r0, v0 = 1D array-like of initial position and velocity in both x and y
             assumming r0[0] is x0 and r0[1] = y0. Same for v0.
    dt = time interval to estimate over
    N = number of steps
    q1, q2 = charges of the particles: electron and ion
    m,k = mass of electron, Couloumb's constant
    """
    # Initialization
    r = np.zeros((N, 2))#col: x-position, y in m
    v = np.zeros((N, 2)) #col: vx, vy in m/s
    a = np.zeros((N,2)) #ax, ay
    t = np.zeros(N) #time in sec
    # Set the first term to initial conditions
    t[0] = t0
    r[0, 0] = r0[0] #x0
    r[0, 1] = r0[1] #y0
    v[0, 0] = v0[0] #v_x0
    v[0, 1] = v0[1] #v_y0
    a[0, :] = q1*q2*k/(m*(r[0,0]**2 + r[0,1]**2)**1.5)*r[0,:]
    for n in range(N-1):
        r[n+1, :] = r[n,:] + dt*v[n,:]
        v[n+1, :] = v[n,:] + dt*a[n,:]
        a[n+1, :] = q1*q2*k/(m*(r[n+1,0]**2 + r[n+1,1]**2)**1.5)*r[n+1,:]
        t[n+1] = (n+1)*dt
    return t, r, v,a

# Setting up initial conditions
r0 = np.array([x0, y0]) #in m
v0 = np.array([v0x, v0y]) #m/s
t0 = 0 #sec
dt = 1e-17 #sec
N = 5000
t, r, v,a = euler(t0, r0, v0, dt, N, z*q, -q, m, k)
# net magnitude of position, velocity, and acceleration
pos = np.sqrt(r[:,0]**2 + r[:,1]**2)
vel = np.sqrt(v[:,0]**2 + v[:,1]**2)
accel= np.sqrt(a[:,0]**2 + a[:,1]**2)

###############################################################################
########################### Power Spectrum ####################################
def getpower(t, s, v, dt, n, q1, q2, m, k, mode = 'Power'):
    tp, rp, vp, ap = euler(t, s, v, dt, N, q1, q2, m, k)
    # Get a_net = net acceleration
    a_tot = np.sqrt(ap[:,0]**2 + ap[:,1]**2)
    ## Fourier Transform the acceleration
    a_ft = np.fft.fft(a_tot)
    # Get the associated freq of length n for a time step dt
    f = np.fft.fftfreq(n, dt) #in Hz
    # sort the freq bc f is given as 0 to +, the - to 0, so the alignment if off
    # need to get this series of indices so can use them for plotting the power
    # spectrum later on
    ind = np.argsort(f)
    power = 2*q**2*abs(a_ft)**2/(3*c**3) #need to use abs(), np.conjugate doesn't work
#    plt.figure(figorder)
#    plt.plot(rp[:,0]/a0, rp[:,1]/a0)
#    plt.xlabel(r'x ($a_0$)')
#    plt.ylabel(r'y ($a_0$)')
#    plt.xlim(s[0]/a0, -s[0]/a0+100)
#    plt.ylim(s[1]/a0-1, s[1]/a0+.1)
#    plt.show()
    if mode != 'Power':
        return tp, rp, vp, ap, f, power
    else:     
        return ind, f, power
    
ind, freq ,P_net = getpower(t0, r0, v0, dt, N, z*q, -q, m, k)
###############################################################################
########################### Different b and v0 ################################
b = [0.01*y0, 0.1*y0, 10*y0, 50*y0] #different impact parameters
v0_i = [0.01*v0, 0.1*v0, 10*v0, 50*v0] #different initial velocities
sb = len(b)
# vary b only:
sb3= np.array([x0, b[3]])
sb2= np.array([x0, b[2]])
indb3, fb3, pb3 = getpower(t0, sb3, v0, dt, N, z*q, -q, m, k)
indb2, fb2, pb2 = getpower(t0, sb2, v0, dt, N, z*q, -q, m, k)
# vary v0 only:
indv2, fv2, pv2 = getpower(t0, r0, v0_i[2], dt, N, z*q, -q, m, k)
indv3, fv3, pv3 = getpower(t0, r0, v0_i[3], dt, N, z*q, -q, m, k)

###############################################################################
############################## Plotting #######################################
# position
f,(f1,f2,f3)= plt.subplots(3,sharex = True)
f1.plot(t*1e18, r[:, 0]/a0)
f1.set_ylabel(r'x $(a_0)$')
f2.plot(t*1e18, r[:, 1]/a0)
f2.set_ylabel(r'y $(a_0)$')
f3.plot(t*1e18, pos/a0)
f3.set_ylabel(r'$r (a_0)$')
f3.set_xlabel('Time (attosec)')
f.suptitle(r'x-, y-position, and net position of the electron in unit of Bohr radius, $a_0$' )
# velocity
g,(g1,g2,g3)=plt.subplots(3, sharex = True)
g1.plot(t*1e18, v[:, 0])
g1.set_ylabel(r'$v_x$ (m/s)')
g2.plot(t*1e18, v[:, 1])
g2.set_ylabel(r'$v_y$ (m/s)')
g3.plot(t*1e18, vel)
g3.set_xlabel('Time (as)')
g3.set_ylabel(r'$v_{net}$ (m/s)')
g.suptitle(r'$v_x$, $v_y$, and $|v_{net}|$ of the electron' )
g.subplots_adjust(top=0.88, left = 0.2) 
# acceleration
h,(h1,h2,h3)=plt.subplots(3, sharex = True)
h1.plot(t*1e18, a[:, 0])
h1.set_ylabel(r'$a_x (m/s^2)$')
h2.plot(t*1e18, a[:, 1])
h2.set_ylabel(r'$a_y (m/s^2)$')
h3.plot(t*1e18, -accel)
h3.set_ylabel(r'$a_{net} (m/s^2)$')
h3.set_xlabel('Time (as)')
h.subplots_adjust(top=0.88, left = 0.15,right = 0.94) 
h.suptitle(r'$a_x, a_y$, and $a_{net}$ of electron')
# x vs y
plt.figure(4)
plt.title('y vs x position of the electron in unit of $a_0$')
plt.plot(r[:,0]/a0, r[:,1]/a0)
plt.xlabel(r'x ($a_0$)')
plt.ylabel(r'y ($a_0$)')
plt.xlim(x0/a0, -x0/a0+100)
# Power spectrum
plt.figure(5)
plt.plot(freq[ind], P_net[ind], 'b-')
plt.yscale('log')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power Radiated')
# Experiment with Different parameters
plt.figure(6)
plt.plot(freq[ind], P_net[ind], 'k-', label = r'$b_0 = 300a_0,v_{0x} = 0.005c$')
plt.plot(fb2[indb2], pb2[indb2], 'r--',label = r'b = $10b_0,v_{init,x} = v_{0x}$')
plt.plot(fb3[indb3], pb3[indb3], 'm--',label = r'b = $50b_0,v_{init,x} = v_{0x}$')
plt.plot(fv2[indv2], pv2[indv2], 'c--',label = r'b = $b_0,v_{init,x} = 10v_{0x}$')
plt.plot(fv3[indv3], pv3[indv3], 'b--',label = r'b = $b_0,v_{init,x} = 500v_{0x}$')
plt.yscale('log')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power Radiated')
plt.legend()

plt.show()
