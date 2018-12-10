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
            getpower(time, s, vel, dtp, n, q1, q2, m, k, mode = 'Power')- FFT a and 
                compute a power spectrum.
                time, s, vel, dtp, n, q1, q2, m, k= args for calling the euler fn
                mode = default to 'Power'-> only return the necessary components
                for plotting the power spectrum wrt to freq.
            getpeak(frequency,Pspec) - get the peak Power and the associated freq
                
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
y0 = 1e3*a0 #initial position of the electron away from the ion in angstrom
x0 = -1e4*a0
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
dt = 1e-15 #sec
N = 1000
t, r, v,a = euler(t0, r0, v0, dt, N, z*q, -q, m, k)
# net magnitude of position, velocity, and acceleration
pos = np.sqrt(r[:,0]**2 + r[:,1]**2)
vel = np.sqrt(v[:,0]**2 + v[:,1]**2)
accel= np.sqrt(a[:,0]**2 + a[:,1]**2)

###############################################################################
########################### Power Spectrum ####################################
def getpower(time, s, vel, dtp, n, q1, q2, m, k, mode = 'Power'):
    tp, rp, vp, ap = euler(time, s, vel, dtp, n, q1, q2, m, k)
    # FT the x and y-component of acceleration
    ax_ft = np.fft.fft(ap[:,0])
    ay_ft = np.fft.fft(ap[:,1])
    #angle between them
    theta = np.arctan(rp[:,1]/rp[:,0])
    # get only the radial component of the acceleration by dotting a_vector with r_hat
    ar_ft = ax_ft*np.cos(theta) + ay_ft*np.sin(theta)
    # Get the associated freq of length n for a time step dt
    f = np.fft.fftfreq(n, dtp) #in Hz
    # sort the freq bc f is given as 0 to +, the - to 0, so the alignment if off
    # need to get this series of indices so can use them for plotting the power
    # spectrum later on
    ind = np.argsort(f)
    power = 2*q**2*np.abs(ar_ft)**2/(3*c**3) #need to use abs(), np.conjugate doesn't work
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
b = np.array([0.5*y0, 0.8*y0, y0, 10*y0, 20*y0, 30*y0]) #different impact parameters
v0_i = np.array([0.5*v0, v0, 10*v0, 20*v0, 30*v0]) #different initial velocities
sb = len(b)
## vary b only:
dt1 = 1e-14
dt3 = 1e-13
sb0= np.array([x0, b[0]])
sb1 = np.array([x0, b[1]])
sb3= np.array([x0, b[3]])
sb4 = np.array([x0, b[4]])
sb5 = np.array([x0, b[5]])
indb0, fb0, pb0 = getpower(t0, sb0, v0, dt,N, z*q, -q, m, k)
indb1, fb1, pb1 = getpower(t0, sb1, v0, dt,N, z*q, -q, m, k)
indb3, fb3, pb3 = getpower(t0, sb3, v0, dt1,N, z*q, -q, m, k)
indb4, fb4, pb4 = getpower(t0, sb4, v0, dt3,N, z*q, -q, m, k)
indb5, fb5, pb5 = getpower(t0, sb5, v0, dt3,N, z*q, -q, m, k)
## vary v0 only:
indv0, fv0, pv0 = getpower(t0, r0, v0_i[0], dt1, N, z*q, -q, m, k)
indv2, fv2, pv2 = getpower(t0, r0, v0_i[2], dt, N, z*q, -q, m, k)
indv3, fv3, pv3 = getpower(t0, r0, v0_i[3], dt, N, z*q, -q, m, k)
indv4, fv4, pv4 = getpower(t0, r0, v0_i[4], dt, N, z*q, -q, m, k)

#### Finding the peak:
def getpeak(frequency,Pspec):
    s = len(Pspec)
    for i in range(s//2): 
    ##only half of the spectrum bc did the FFT so there's 
    #negative freq, and we only care about positive freq
        if Pspec[i] == max(Pspec):
            return i, frequency[i], Pspec[i]
## vary b only:    
i0b, fpeak0b, Ppeak0b = getpeak(fb0, pb0)
i1b, fpeak1b, Ppeak1b = getpeak(fb1, pb1)
i0, fpeak0, Ppeak0 = getpeak(freq,P_net) 
i3b, fpeak3b, Ppeak3b = getpeak(fb3, pb3)
i4b, fpeak4b, Ppeak4b = getpeak(fb4, pb4)
i5b, fpeak5b, Ppeak5b = getpeak(fb5, pb5)
## vary v0 only 
i0v, fpeak0v, Ppeak0v = getpeak(fv0, pv0)
i2v, fpeak2v, Ppeak2v = getpeak(fv2, pv2)
i3v, fpeak3v, Ppeak3v = getpeak(fv3, pv3)
i4v, fpeak4v, Ppeak4v = getpeak(fv4, pv4)
fpeakb = np.array([fpeak0b,fpeak1b, fpeak0, fpeak3b, fpeak4b, fpeak5b])
fpeakv = np.array([fpeak0v, fpeak0, fpeak2v, fpeak3v, fpeak4v])

## fitting
def fit(x, a2, a1, a0):
    return a0+ a1*x + a2*(x**2)#+ a3*(x**3) +a4*(x**4)+ a5*(x**5)+ a6*(x**6)

pb= np.polyfit(b, fpeakb,2)
b_fit = np.linspace(b[0], b[len(b) -1], 50)
fb_fit = fit(b_fit, pb[0], pb[1], pb[2])

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
f.subplots_adjust(top=0.88, left = 0.2) 
# velocity
g,(g1,g2,g3)=plt.subplots(3, sharex = True)
g1.plot(t*1e18, v[:, 0])
g1.set_ylabel(r'$v_x$ (m/s)')
g2.plot(t*1e18, v[:, 1])
g2.set_ylabel(r'$v_y$ (m/s)')
g3.plot(t*1e18, vel)
g3.set_xlabel('Time (as)')
g3.set_ylabel(r'$|v_{net}|$ (m/s)')
g.suptitle(r'$v_x$, $v_y$, and $|v_{net}|$ of the electron' )
g.subplots_adjust(top=0.88, left = 0.2) 
# acceleration
h,(h1,h2,h3)=plt.subplots(3, sharex = True)
h1.plot(t*1e18, a[:, 0])
h1.set_ylabel(r'$a_x (m/s^2)$')
h2.plot(t*1e18, a[:, 1])
h2.set_ylabel(r'$a_y (m/s^2)$')
h3.plot(t*1e18, accel) 
h3.set_ylabel(r'$|a_{net}| (m/s^2)$')
h3.set_xlabel('Time (as)')
h.subplots_adjust(top=0.88, left = 0.15,right = 0.94) 
h.suptitle(r'$a_x, a_y$, and $|a_{net}|$ of electron')
# x vs y
plt.figure(4)
plt.title('y vs x position of the electron in unit of $a_0$')
plt.plot(r[:,0]/a0, r[:,1]/a0)
plt.xlabel(r'x ($a_0$)')
plt.ylabel(r'y ($a_0$)')
plt.xlim(x0/a0, -x0/a0+100)
# Power spectrum
plt.figure(5)
plt.plot(freq[ind], P_net[ind], 'b-', label = r'$b_0 = 1000a_0,v_{0x} = 0.005c$')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power Radiated')
plt.xlim(0, 4e13)
plt.legend()
# Experiment with Different parameters
p, (p1, p2, p3) = plt.subplots(3)
p1.plot(fb3[indb3], pb3[indb3], 'r--',label = r'b = $10b_0,v_{init,x} = v_{0x}$')
p1.set_xlim(-1e10, 2e13)
p1.legend()
p1.set_ylabel('Power Radiated')
p2.plot(fv3[indv3], pv3[indv3], 'b--',label = r'b = $b_0,v_{init,x} = 500v_{0x}$')
p2.set_ylabel('Power Radiated')
p2.set_xlim(0, 3e13)
p2.legend()
plt.plot(fv2[indv2], pv2[indv2], 'c--',label = r'b = $b_0,v_{init,x} = 10v_{0x}$')
p3.set_xlim(0, 2e14)
p3.set_ylabel('Power Radiated')
p3.legend()
p3.set_xlabel('Frequency (Hz)')
p.suptitle(r'Power Spectrum for different b and v with $b_0 = 1000a_0,v_{0x} = 0.005c$')
p.subplots_adjust(top=0.88, left = 0.2, hspace = 0.5) 
# Frequency vs b/v0
plt.figure(7)
plt.subplot(121)
plt.title('Vary Impact Parameters Only')
plt.plot(b/a0, fpeakb, 'b-')
plt.xlabel(r'b ($a_0$)')
plt.ylabel('Frequency (Hz)')
plt.subplot(122)
plt.title(r'Vary $v_{0,x}$ Only. $v_{0,x}$ is in Unit of c')
plt.plot(v0_i[:,0]/c, fpeakv, 'r-')
plt.xlabel(r'$v_{0,x}$ (c)')
plt.ylabel('Frequency (Hz)')


plt.show()
