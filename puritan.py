#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 13:22:36 2024

@author: cjawetz3
"""

import numpy as np
from qiskit import QuantumCircuit, transpile, assemble, QuantumRegister, AncillaRegister
from qiskit_aer import Aer
from qiskit.circuit.library import Permutation
from qiskit.quantum_info.operators import Operator
from qiskit.quantum_info import Statevector, StabilizerState, Clifford
from qiskit.visualization import array_to_latex, plot_histogram, plot_bloch_multivector
from qiskit.result import marginal_counts
import time, math
from math import pi, sqrt
import pickle
from collections import Counter
import scipy.io as spio
import qiskit.result
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from pdf2image import convert_from_path
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib
from scipy.optimize import fsolve
from scipy import special
from scipy.stats import linregress
import pandas as pd
from scipy.interpolate import interp1d

plt.rc('axes', labelsize=9.)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=9.)    # fontsize of the tick labels
plt.rc('ytick', labelsize=9.)    # fontsize of the tick labels
plt.rc('legend', fontsize=9.)    # fontsize of the tick labels
# rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
# rc('text', usetex=True)

# D1Q3 lattice constants
D = 1/9    #diffusion constant
w = np.array([1/6, 2/3, 1/6]) # weight coeffecients
cx = np.array([1, 0, -1])   #lattice velocities
csq = 1/3   #square of sound speed
ux = 0.  # advection x-velocity
tm=0.4
cp=1
la=10
tb=1
to=0
dx=1
dt=1
#D = 0.15 # diffusion coefficient (equivalent to viscosity)
tau = D/csq + 1/2 #relaxation time
omega = 1/tau #2.0/(6*alpha/csq^2+dt)
alpha=(1/6)*(2-dt)
#alpha=0.08
J=1
k_b=1.38*(10^(-23))
def f(lambd):
    val1=cp*(tb-tm)/la/(np.exp(lambd**2)*special.erf(lambd))
    val2=(cp*(tm-to)/la)/(np.exp(lambd**2)*special.erfc(lambd))
    val3=lambd*np.sqrt(np.pi)
    return val1-val2-val3


def compute_feq(rho, w, cx, ux, csq):
    feq = np.zeros((3,M))
    for i in range(3):
        feq[i] = w[i] * (1 + cx[i]*ux/csq) * rho
    return feq

def ini(x, w, cx, ux, csq):
    M = len(x)
    rho = np.zeros(M)    #Delta function as initial density
    rho[0] = 1
    feq = compute_feq(rho, w, cx, ux, csq)
    f = feq
    state=np.zeros(M)
    state[0]=1
    return f, rho, state

def update_encoding(qc, f, M):
    for k in range(M): 
        amp000 = np.sqrt( (1-f[0][k]) * (1-f[1][k])* (1-f[2][k]) )
        amp100 = np.sqrt( (1-f[1][k]) * (1-f[2][k]) * f[0][k] )
        amp010 = np.sqrt( (1-f[0][k]) * (1-f[2][k]) * f[1][k] )        
        amp001 = np.sqrt( (1-f[0][k]) * (1-f[1][k]) * f[2][k] )
        amp011 = np.sqrt( (1-f[0][k]) * f[1][k] * f[2][k] )
        amp101 = np.sqrt( (1-f[1][k]) * f[0][k] * f[2][k] )
        amp110 = np.sqrt( (1-f[2][k]) * f[0][k] * f[1][k] )
        amp111 = np.sqrt( f[0][k] * f[1][k] * f[2][k] )
        vector = np.array([amp111, amp110, amp101, amp011, amp001, amp010, amp100, amp000])
        qc.prepare_state(vector, [0+3*k, 1+3*k, 2+3*k])  
    return qc

def update_encoding_rate(qc, f, M):
    for k in range(M-rate): 
        amp000 = np.sqrt( (1-f[0][k]) * (1-f[1][k])* (1-f[2][k]) )
        amp100 = np.sqrt( (1-f[1][k]) * (1-f[2][k]) * f[0][k] )
        amp010 = np.sqrt( (1-f[0][k]) * (1-f[2][k]) * f[1][k] )        
        amp001 = np.sqrt( (1-f[0][k]) * (1-f[1][k]) * f[2][k] )
        amp011 = np.sqrt( (1-f[0][k]) * f[1][k] * f[2][k] )
        amp101 = np.sqrt( (1-f[1][k]) * f[0][k] * f[2][k] )
        amp110 = np.sqrt( (1-f[2][k]) * f[0][k] * f[1][k] )
        amp111 = np.sqrt( f[0][k] * f[1][k] * f[2][k] )
        vector = np.array([amp111, amp110, amp101, amp011, amp001, amp010, amp100, amp000])
        qc.prepare_state(vector, [0+3*k, 1+3*k, 2+3*k])  
    return qc

U = [[0, 0, 0, 0, 0, 0, 0, 1],
     [0, 0, 0, 0, 1j/sqrt(3), np.exp(-1j*pi/6)/sqrt(3), np.exp(-1j*pi/6)/sqrt(3), 0],
     [0, 0, 0, 0, np.exp(-1j*pi/6)/sqrt(3), 1j/sqrt(3), np.exp(-1j*pi/6)/sqrt(3), 0],
     [0, np.exp(-1j*pi/6)/sqrt(3), np.exp(-1j*pi/6)/sqrt(3), 1j/sqrt(3), 0, 0, 0, 0],
     [0, 0, 0, 0, np.exp(-1j*pi/6)/sqrt(3), np.exp(-1j*pi/6)/sqrt(3), 1j/sqrt(3), 0],
     [0, np.exp(-1j*pi/6)/sqrt(3), 1j/sqrt(3), np.exp(-1j*pi/6)/sqrt(3), 0, 0, 0, 0],
     [0, 1j/sqrt(3), np.exp(-1j*pi/6)/sqrt(3), np.exp(-1j*pi/6)/sqrt(3), 0, 0, 0, 0],
     [1, 0, 0, 0, 0, 0, 0, 0]]


def collision_Diffusion(circ, k):
    circ.unitary(U, [0+3*k,1+3*k,2+3*k])
    return circ

def classical_streaming_map(n):
    """
    Compute the mapping of bit positions after applying the streaming pattern.
    
    :param n: Total number of qubits (should be a multiple of 3)
    :return: A dictionary mapping original positions to new positions
    """
    if n < 6 or n % 3 != 0:
        raise ValueError("n must be at least 6 and a multiple of 3")

    # Initialize the mapping with identity (no change)
    mapping = {i: i for i in range(n)}
    
    # Apply the streaming pattern
    pattern = computeStreamingPattern(n)
    for i in range(0, len(pattern), 3):
        a, b, c = pattern[i:i+3]
        # Swap the mappings
        #mapping[a], mapping[b], mapping[c] = mapping[b], mapping[c], mapping[a]
    
    return mapping

def get_sum_qubit_indices(n, original_indices):
    """
    Get the new indices of the qubits we want to sum over after streaming.
    
    :param n: Total number of qubits
    :param original_indices: List of original qubit indices we want to sum
    :return: List of new qubit indices to sum over
    """
    mapping = classical_streaming_map(n)
    return [mapping[i] for i in original_indices]


def computeStreamingPattern(n):
    if (n >= 6):  #minimum for streaming is 2 sites, corresonding to 6qubits
        #first pair of qubits
        streamingPattern = [2, 1, 5]
        for i in range(3,n-4):
            if i%3 == 0:  
                streamingPattern.extend([i-3, i+1, i+5])
        #last pair of qubits
        streamingPattern.extend([n-6, n-2, n-3])
    else:
        streamingPattern = []
    return streamingPattern

def classical_streaming(M, state, qubit_counts, numberOfShots):
    #read post collision probabilities
    f = np.zeros((3,M))
    fout = np.zeros((3,M))
    for k in range(M):
        if '1' in qubit_counts[2+3*k]:
            fout[0][k] = qubit_counts[2+3*k]['1'] / numberOfShots
        if '1' in qubit_counts[1+3*k]:
            fout[1][k] = qubit_counts[1+3*k]['1'] / numberOfShots
        if '1' in qubit_counts[0+3*k]:
            fout[2][k] = qubit_counts[0+3*k]['1'] / numberOfShots
    
    # classical streaming
    for k in range(1,M):    #right 
        f[0][k] = fout[0][k-1]

    for k in range(M):     #stationary 
        f[1][k] = fout[1][k] 
        
    for k in range(M-1):     #left
        f[2][k] = fout[2][k+1]

    for i in range(M):
            if np.sum(f,axis=0)[i]>tm:
                if state[i]<1:
                    state[i]+=(np.sum(f,axis=0)[i]-tm)*cp/la
                    column_sum=(np.sum(f,axis=0)[i])                    
                    f[0][i]*=tm/column_sum
                    f[1][i]*=tm/column_sum
                    f[2][i]*=tm/column_sum
                if state[i]>=1:
                    column_sum=(np.sum(f,axis=0)[i])
                    f[0][i]+=f[0][i]*(state[i]-1)*la/cp/column_sum
                    f[1][i]+=f[1][i]*(state[i]-1)*la/cp/column_sum
                    f[2][i]+=f[2][i]*(state[i]-1)*la/cp/column_sum
                    state[i]=1
#      periodic BC
    #f[0][0] = f[0][M-1]
    #f[2][M-1] = f[2][0]
    f[0][0] = 1-f[2][0]-f[1][0]
    f[0][M-1]=0#fout[0][M-1]
    f[1][M-1]=0#fout[0][M-1]
    f[2][M-1]=0#fout[0][M-1]
    #f[1][0] = 1-f[0][0]
    #f[2][0] = 0
    
    #periodic BC
    # f[0][0] = f[0][M-1]
    # f[2][M-1] = f[2][0]

    return f,state

def oneTimeStep_classicalStreaming(f, M, state, numberOfShots, backend):
    qc = QuantumCircuit(3*M)
    #step1: encoding
    qc = update_encoding(qc, f, M)
    qc.barrier()
    
    #step2: collision
    for k in range(M):
        qc = collision_Diffusion(qc, k)
        
    #step3: measurement
    qc.measure_all()
    job = transpile(qc, backend)
    result=backend.run(job, shots=numberOfShots).result()
    counts = result.get_counts(0)
    qubit_counts = [marginal_counts(counts, [qubit]) for qubit in range(3*M)]

    #step4: streaming
    f,state = classical_streaming(M, state, qubit_counts, numberOfShots)
    return f, qc,state

def oneTimeStep_quantumStreaming(f, M, numberOfShots, backend, t, maxT,bc,bcl,bcr):
    qc = QuantumCircuit(3*M+rate)
    #step1: encoding
    qc = update_encoding_rate(qc, f, M+rate)
    qc.barrier()
    
    #excess heat ancilla
    # qc.cry(2*np.pi/3,3*M-1,3*M)
    # qc.cry(2*np.pi/3,3*M-2,3*M)
    # qc.cry(2*np.pi/3,3*M-3,3*M)
    
    #step2: collision
    for k in range(M):
        qc = collision_Diffusion(qc, k)
        
    
    
    #step3: streaming    
    qc.append(Permutation(num_qubits = 3*M, pattern = computeStreamingPattern(3*M)), range(3*M))
    
    # Get new indices to sum over
    original_indices = [0, 1, 2]  # Assuming these were the original qubits to sum
    new_sum_indices = get_sum_qubit_indices(3*M, original_indices)


    #qc.ry(angle,3*M+np.mod(t,rate))
    qc.mcry(2*np.arcsin(np.sqrt(1)),new_sum_indices,3*M+np.mod(t,rate))
    qc.x(new_sum_indices[0])
    qc.mcry(2*np.arcsin(np.sqrt(2/3)),new_sum_indices,3*M+np.mod(t,rate))
    qc.x(new_sum_indices[1])
    qc.mcry(2*np.arcsin(np.sqrt(1/3)),new_sum_indices,3*M+np.mod(t,rate))
    qc.x(new_sum_indices[0])
    qc.mcry(2*np.arcsin(np.sqrt(2/3)),new_sum_indices,3*M+np.mod(t,rate))
    qc.x(new_sum_indices[2])
    qc.mcry(2*np.arcsin(np.sqrt(1/3)),new_sum_indices,3*M+np.mod(t,rate))
    qc.x(new_sum_indices[0])
    qc.mcry(0,new_sum_indices,3*M+np.mod(t,rate))
    qc.x(new_sum_indices[1])
    qc.mcry(2*np.arcsin(np.sqrt(1/3)),new_sum_indices,3*M+np.mod(t,rate))
    qc.x(new_sum_indices[0])
    qc.mcry(2*np.arcsin(np.sqrt(2/3)),new_sum_indices,3*M+np.mod(t,rate))
    qc.x(new_sum_indices[2])

    # if np.mod(t,rate)!=0 and t>rate:
    #     qc.reset(3*M-1)
    #     qc.ry(2*math.asin(bcr[2]*np.pi),0)
    
    #     qc.reset(0)
    #     qc.ry(2*math.asin(np.sqrt(bcl[2])),0)
    #     qc.reset(1)
    #     qc.ry(2*math.asin(np.sqrt(bcl[1])),1)
    #     qc.reset(2)
    #     qc.ry(2*math.asin(np.sqrt(bcl[0])),2)

    qc.measure_all()
    job = transpile(qc, backend)
    result=backend.run(job, shots=numberOfShots).result()
    counts = result.get_counts(0)
    qubit_counts = [marginal_counts(counts, [qubit]) for qubit in range(3*M)]
    ex_counts=marginal_counts(counts, indices=[int(3*M+np.mod(t,rate))])
    #ex_q=(7.194*(ex_counts['1']/numberOfShots-tm))+2.532
    try:
        ex_q=(ex_counts['1']/numberOfShots)*3-tm#-tm
        #ex_q=68.03*(ex_counts['1']/numberOfShots+0.0009)-tm
        ex_raw=ex_counts['1']/numberOfShots
    except KeyError as e:
        ex_q=-tm
        #ex_q=68.03*(ex_counts['1']/numberOfShots+0.0009)-tm
        ex_raw=0

    #read post streaming probabilities
    fout = np.zeros((3,M))
    for k in range(M):
        if '1' in qubit_counts[2+3*k]:
            fout[0][k] = qubit_counts[2+3*k]['1'] / numberOfShots
        if '1' in qubit_counts[1+3*k]:
            fout[1][k] = qubit_counts[1+3*k]['1'] / numberOfShots
        if '1' in qubit_counts[0+3*k]:
            fout[2][k] = qubit_counts[0+3*k]['1'] / numberOfShots
    f = fout
    ex=f[0][0]+f[1][0]+f[2][0]-tm
    lin_check.append([ex,ex_q])
    # print(ex_q)
    # print(ex)
    #periodic BC
    # f[0][0] = f[0][M-1]
    # f[2][M-1] = f[2][0]
    if np.mod(t,min(rate,short))==0 or t<=2*rate:
        f[2][0] = bc-f[1][0]-f[0][0]
        f[0][M-1] = f[0][M-2]
        f[1][M-1] = 0
        f[2][M-1] = 0
    else:
        f[2][0]=bcl[2]    
        f[1][0]=bcl[1]    
        f[0][0]=bcl[0]    
        f[0][M-1] = f[0][M-2]
        f[1][M-1] = 0
        f[2][M-1] = 0
    #f[1][0] = bc-f[2][0]
    #f[0][0] = 0
    return f, qc, ex_q, ex, ex_raw

lin_check=[]
def oneTimeStep_quantumStreaming_liq(f, M, numberOfShots, backend, t, maxT,bc,bcl,bcr):
    qc = QuantumCircuit(3*M+rate)
    #step1: encoding
    qc = update_encoding_rate(qc, f, M+rate)
    qc.barrier()
    
    #excess heat ancilla
    # qc.cry(2*np.pi/3,3*M-1,3*M)
    # qc.cry(2*np.pi/3,3*M-2,3*M)
    # qc.cry(2*np.pi/3,3*M-3,3*M)
    
    #step2: collision
    for k in range(M):
        qc = collision_Diffusion(qc, k)
        
    
    #theta=0.1
    
    #step3: streaming    
    qc.append(Permutation(num_qubits = 3*M, pattern = computeStreamingPattern(3*M)), range(3*M))
    # Get new indices to sum over
    original_indices = [3*M-3,3*M-2,3*M-1]  # Assuming these were the original qubits to sum
    new_sum_indices = get_sum_qubit_indices(3*M, original_indices)

    #step4: excess heat
    qc.mcry(2*np.arcsin(np.sqrt(1)),new_sum_indices,3*M+np.mod(t,rate))
    qc.x(new_sum_indices[0])
    qc.mcry(2*np.arcsin(np.sqrt(2/3)),new_sum_indices,3*M+np.mod(t,rate))
    qc.x(new_sum_indices[1])
    qc.mcry(2*np.arcsin(np.sqrt(1/3)),new_sum_indices,3*M+np.mod(t,rate))
    qc.x(new_sum_indices[0])
    qc.mcry(2*np.arcsin(np.sqrt(2/3)),new_sum_indices,3*M+np.mod(t,rate))
    qc.x(new_sum_indices[2])
    qc.mcry(2*np.arcsin(np.sqrt(1/3)),new_sum_indices,3*M+np.mod(t,rate))
    qc.x(new_sum_indices[0])
    qc.mcry(0,new_sum_indices,3*M+np.mod(t,rate))
    qc.x(new_sum_indices[1])
    qc.mcry(2*np.arcsin(np.sqrt(1/3)),new_sum_indices,3*M+np.mod(t,rate))
    qc.x(new_sum_indices[0])
    qc.mcry(2*np.arcsin(np.sqrt(2/3)),new_sum_indices,3*M+np.mod(t,rate))
    qc.x(new_sum_indices[2])
    
    # if np.mod(t,rate)!=0 and t>rate:
    #     qc.reset(3*M-1)
    #     qc.ry(2*math.asin(np.sqrt(bcr[0])),3*M-1)
    #     qc.reset(3*M-2)
    #     qc.ry(2*math.asin(np.sqrt(bcr[1])),3*M-2)
    #     qc.reset(3*M-3)
    #     qc.ry(2*math.asin(np.sqrt(bcr[2])),3*M-3)
    #     qc.reset(0)
    #     qc.ry(2*math.asin(np.sqrt(bcl[2])),0)
    #     qc.reset(1)
    #     qc.ry(2*math.asin(np.sqrt(bcl[1])),1)    
    #     qc.reset(2)
    #     qc.ry(2*math.asin(np.sqrt(bcl[0])),2)
    qc.barrier()
    qc.measure_all()
    job = transpile(qc, backend)
    result=backend.run(job, shots=numberOfShots).result()
    counts = result.get_counts(0)
    qubit_counts = [marginal_counts(counts, [qubit]) for qubit in range(3*M)]
    ex_counts=marginal_counts(counts, indices=[int(3*M+np.mod(t,rate))])
    try:
        #ex_q=(7.194*(ex_counts['1']/numberOfShots-tm))+2.532
        #ex_q=68.03*(ex_counts['1']/numberOfShots+0.0009)-tm
        ex_q=(ex_counts['1']/numberOfShots)*3-tm
        ex_raw=ex_counts['1']/numberOfShots
    except KeyError as error:
        ex_q=0
    #read post streaming probabilities
    fout = np.zeros((3,M))
    for k in range(M):
        if '1' in qubit_counts[2+3*k]:
            fout[0][k] = qubit_counts[2+3*k]['1'] / numberOfShots
        if '1' in qubit_counts[1+3*k]:
            fout[1][k] = qubit_counts[1+3*k]['1'] / numberOfShots
        if '1' in qubit_counts[0+3*k]:
            fout[2][k] = qubit_counts[0+3*k]['1'] / numberOfShots
    #if '1' in qubit_counts[3*M]:
    #    ex=3*qubit_counts[3*M]['1'] / numberOfShots - tm
    #print(ex)
    f = fout

    
    ex=f[0][M-1]+f[1][M-1]+f[2][M-1]-tm
    # print(ex_q)
    # print(ex)
    lin_check.append([ex,ex_q])
    if np.mod(t,min(rate,short))==0 or t<=2*rate:
        f[0][M-1] = tm-f[2][M-1]-f[1][M-1]
        # column_sum=f[0][M-1]+f[1][M-1]+f[2][M-1]
        # if column_sum-tm>(-1e-3):
        #     f[2][M-1] *= tm/column_sum
        #     f[1][M-1] *= tm/column_sum
        #     f[0][M-1] *= tm/column_sum
        f[2][0] = bc-f[1][0]-f[0][0]

    else:
        f[0][M-1]=bcr[0]
        f[1][M-1]=bcr[1]
        f[2][M-1]=bcr[2]
        f[0][0]=bcl[0]        
        f[1][0]=bcl[1]        
        f[2][0]=bcl[2]        

    #f[0][M-1] = tm+ex-f[1][M-1]-f[2][M-1]
    
    return f, qc,ex_q, ex, ex_raw


# number operators
n1 = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
              [0, 1, 0, 0, 0, 0, 0, 0],
              [0, 0, 1, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 1, 0],
              [0, 0, 0, 0, 0, 0, 0, 0]])

n2 = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
              [0, 1, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 1, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 1, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0]])

n3 = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 1, 0, 0, 0, 0, 0],
              [0, 0, 0, 1, 0, 0, 0, 0],
              [0, 0, 0, 0, 1, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0]])


 
# collision operator - Diffusion equation
coeff = np.exp(-1j*pi/6)/sqrt(3)
a = np.exp(1j*pi/6)*sqrt(3)
b = np.exp(1j*2*pi/3)
C_Diffusion = coeff *  np.array([[a, 0, 0, 0, 0, 0, 0, 0],
                                  [0, b, 1, 1, 0, 0, 0, 0],
                                  [0, 1, b, 1, 0, 0, 0, 0],
                                  [0, 1, 1, b, 0, 0, 0, 0],
                                  [0, 0, 0, 0, b, 1, 1, 0],
                                  [0, 0, 0, 0, 1, b, 1, 0],
                                  [0, 0, 0, 0, 1, 1, b, 0],
                                  [0, 0, 0, 0, 0, 0, 0, a]])



def classicalOneTimeStep(f, state, M,t,rate):
#     initial combined state
    initial_state = np.zeros((M, 8))
    for i in range(M):
        initial_state[i][0] = np.sqrt( f[0][i] * f[1][i] * f[2][i] )
        initial_state[i][1] = np.sqrt( (1-f[2][i]) * f[0][i]  * f[1][i] )
        initial_state[i][2] = np.sqrt( (1-f[1][i]) * f[0][i]  * f[2][i] )
        initial_state[i][3] = np.sqrt( (1-f[0][i]) * f[2][i]  * f[1][i] )
        initial_state[i][4] = np.sqrt( (1-f[0][i]) * (1-f[1][i]) * f[2][i] )
        initial_state[i][5] = np.sqrt( (1-f[0][i]) * (1-f[2][i]) * f[1][i] )
        initial_state[i][6] = np.sqrt( (1-f[1][i]) * (1-f[2][i]) * f[0][i] )
        initial_state[i][7] = np.sqrt( (1-f[0][i]) * (1-f[1][i]) * (1-f[2][i]) )
        
#     post-collision state (8xlattice_sites)
    post_collision_state = C_Diffusion.dot(initial_state.conjugate().transpose())
    
   # post-collision distribution
    post_collision_distribution = np.zeros((3, M))
    for i in range(M):
        post_collision_distribution[0][i] = post_collision_state.conjugate().transpose()[i].dot(n1.dot( post_collision_state.transpose()[i]))
        post_collision_distribution[1][i] = post_collision_state.conjugate().transpose()[i].dot(n2.dot( post_collision_state.transpose()[i]))
        post_collision_distribution[2][i] = post_collision_state.conjugate().transpose()[i].dot(n3.dot( post_collision_state.transpose()[i]))

#     STREAMING STEP 
    f[0][1:M] = post_collision_distribution[0][0:M-1]
    f[1][0:M] = post_collision_distribution[1][0:M]
    f[2][0:M-1] = post_collision_distribution[2][1:M]
    #rate=3
    # if np.mod(t,rate)==2 or t<3:
    #     eff=rate/10
    
#      periodic BC
    #f[0][0] = f[0][M-1]
    #f[2][M-1] = f[2][0]
    f[0][0] = 1-f[2][0]-f[1][0]
    #f[0][M-1] = 0#post_collision_distribution[0][M-1]
    #f[1][M-1] = 0#post_collision_distribution[0][M-1]
    f[2][M-1] = 0#post_collision_distribution[2][M-1]
    #f[1][0] = 1-f[0][0]
    #f[2][0] = 0
    for i in range(M):
            if np.sum(f,axis=0)[i]>tm:
                if state[i]<1:
                    state[i]+=(np.sum(f,axis=0)[i]-tm)*cp/la
                    column_sum = np.sum(f[:, i])
                    if column_sum != 0:  # To avoid division by zero
                        f[:, i] *= tm / column_sum

                if state[i]>=1:
                    column_sum = np.sum(f[:, i])                    
                    f[0][i]+=f[0][i]*(state[i]-1)*la/cp/column_sum
                    f[1][i]+=f[1][i]*(state[i]-1)*la/cp/column_sum
                    f[2][i]+=f[2][i]*(state[i]-1)*la/cp/column_sum
                    state[i]=1
    
    return f, state

# def ising_model(rho):
#     H(rho)=-J(Sigma(sigma*sigma))
#     if state(i)>0:

#     exp(-beta*H)    
#     do mqmc, using ising model for phi
#     run at end of timestep
    

L = 16  # domain length 
M = L+1  # number of lattice sites (from 0 to L)
n = 3*M   # number of qubits
x = np.array(range(M)) # 1D lattice
rate=12
#initial condition: delta function 
mu0 = int(np.ceil(L/2)) # mean

numberOfShots = 16384  # number of shots
maxT = 110   # number of time steps

# choose simulation backend
backend = 'aer_simulator_matrix_product_state' 
#backend = 'aer_simulator'#'_statevector' 


#initialization
fini, rho,state = ini(x, w, cx, ux, csq)    

# plt.plot(x, rho)

qc = QuantumCircuit(3*M)
qc = update_encoding(qc, fini, M)
fq_classical = fini
fq_quantum = fini
#fq_quantum=np.loadtxt("fq_quantum.txt")
fClassical = fini
state_cc=np.copy(state)
state_qc=np.copy(state)
lam=fsolve(f,1)
gam=lam[0]
hal=int((M-1)/2)
fq_quantum_sol=fini
fq_quantum_liq=fini
q_state=np.copy(state)
#q_state=np.loadtxt("q_state.txt")
full_liq=0
ch_liq=1
ch_sol=0
full_sol=0
ex=0

# fq_quantum_tot_skip=np.zeros((3,M,maxT))
# fq_quantum_tot=np.zeros((3,M,maxT))
# q_state_tot_skip=np.zeros((M,maxT))
# q_state_tot=np.zeros((M,maxT))
# ex_tot_skip=np.zeros((maxT,1))
# ex_tot=np.zeros((maxT,1))
ex=0
rhoq_quantum = fq_quantum[0] + fq_quantum[1] + fq_quantum[2]
runsum=0
runsum_q=0
n_terms=np.arange(1,200)
# Initialize storage for temperature data
time_series = []
temperature_at_point_liq = []
temperature_at_point_sol = []
temperature_at_point_liq_c = []
temperature_at_point_sol_c = []
temperature_at_point_liq_a = []
temperature_at_point_sol_a = []
q_series=[]
qq_state=[]
cc_state=[]
loc_mod=[]
num_nodes_to_track = 3
qq_states = [[] for _ in range(1,num_nodes_to_track)]
cc_states = [[] for _ in range(1,num_nodes_to_track)]

qq_interp_all=[]
cc_interp_all=[]
rho_q_all=[]
rho_c_all=[]
x_base_all=[]
x_tot_all=[]

# In your simulation loop:
for i in range(num_nodes_to_track-1):
    qq_states[i].append(q_state[i+1])
    cc_states[i].append(state_cc[i+1])


# Choose specific points to monitor (for example, midpoint of liquid and solid regions)
point_liq = int(1)  # Choose a point in the liquid region
point_sol = int(4)  # Choose a point in the solid region
all_fq_liq=[]
all_fq_sol=[]
all_q_state=[]
# Define an exponential function for fitting
def exponential_func(t, a, b, c):
    return a * np.exp(b * t) + c
def interpolate_x_tot(x_base_points, x_tot_points, x_base_query):
    """
    Interpolate x_tot values for given x_base query points.
    
    Parameters:
    -----------
    x_base_points : array-like
        Original x_base values (concatenated x_base_liq and x_base_sol)
    x_tot_points : array-like
        Original x_tot values (concatenated x_liq and x_sol)
    x_base_query : float or array-like
        The x_base value(s) at which to interpolate
        
    Returns:
    --------
    float or array-like
        Interpolated x_tot value(s) at the query point(s)
    """
    # Create interpolation function
    f = interp1d(x_base_points, x_tot_points, 
                 kind='linear',  # You can change to 'cubic' for smoother interpolation
                 bounds_error=False,  # Return nan for points outside bounds
                 fill_value=np.nan)
    
    # Return interpolated value(s)
    return f(x_base_query)
# Store previous boundary values
left_boundary_values = [[], [], []]
mid_boundary_values = [[], [], []]
right_boundary_values = [[], [], []]
left_boundary_values_c = [[], [], []]
mid_boundary_values_c = [[], [], []]
right_boundary_values_c = [[], [], []]
short=rate+1
count=3
ch_old=0
ex_plus=0
ex_minus=0
liq_frac=state[0]
errs=[]

def heat_transfer_interp(t, dt, alpha, gam, L, M, tb, tm, to, n_terms, qq_interp):
    # Calculate original points and values
    loc = 2 * gam * np.sqrt(alpha * t * dt)
    key_num = int((M + 1) / 2)
    
    # Original spatial points
    x_base_liq = np.linspace(0, loc, int((M + 1) / 2))
    x_base_sol = np.linspace(loc, L, key_num)
    x_base = np.concatenate((x_base_liq, x_base_sol[1:]))
    
    # Calculate temperatures at original points
    temps_liq = np.zeros(int((M + 1) / 2))
    temps_sol = np.zeros(key_num)
    
    # Liquid region calculation
    for i in range(int((M + 1) / 2)):
        temps_liq[i] = tb - (tb - tm) * (special.erf((i) * loc / ((M + 1) / 2 - 1) / (2 * np.sqrt(alpha * t * dt)))) / special.erf(gam)
    
    # Solid region calculation
    for i in range(key_num):
        temps_sol[i] = tm + (to - tm) * (x_base_sol[i]) / L + np.sum(
            (2 / (n_terms * np.pi)) * (to - tm) * np.sin(n_terms * np.pi * (x_base_sol[i]) / L) *
            np.exp(-alpha * (n_terms * np.pi / L)**2 * t)
        )
    
    # Combine temperatures
    temps = np.concatenate((temps_liq, temps_sol[1:]))
    
    # Create interpolation function
    temp_interp = interp1d(x_base, temps, kind='linear', fill_value='extrapolate')
    
    return temp_interp(qq_interp)
    
    
for t in range(maxT+1):
    print('t = ', t)
    #quantum implementation, classical streaming
    fq_classical, qc1, state_qc = oneTimeStep_classicalStreaming(fq_classical, M, state_qc, numberOfShots, Aer.get_backend(backend)) 
    rhoq_classical = fq_classical[0] + fq_classical[1] + fq_classical[2]
    #quantum implementation, quantum streaming
    # if t<3:
    #     fq_quantum, qc2 = oneTimeStep_quantumStreaming(fq_quantum, M, numberOfShots, Aer.get_backend(backend), t, maxT,1)
    #     rhoq_quantum = fq_quantum[0] + fq_quantum[1] + fq_quantum[2]
    if not any(0 < x < 1 for x in q_state[1:]):

        fq_quantum, qc2, exc,exc2, ex_raw_s = oneTimeStep_quantumStreaming(fq_quantum, M, numberOfShots, Aer.get_backend(backend), t, maxT,1,fq_quantum[:,0]/rhoq_quantum[0],fq_quantum[:,-1])
        rhoq_quantum = fq_quantum[0] + fq_quantum[1] + fq_quantum[2]
    else:
        if np.mod(t,min(rate,short))==1 or t<2*rate or min(rate,short)==1:        
            fq_quantum_liq=np.copy(fq_quantum)
            fq_quantum_liq=fq_quantum_liq[:,:ch_liq+1]
            #for i in range(M):
                # if i==ch_liq+1:
                #     fq_quantum_liq[0][i]=fq_quantum[0][i]*tm/rhoq_quantum[i]
                #     fq_quantum_liq[1][i]=fq_quantum[1][i]*tm/rhoq_quantum[i]
                #     fq_quantum_liq[2][i]=fq_quantum[2][i]*tm/rhoq_quantum[i]
                # if i>=ch_liq+1:
                #     fq_quantum_liq[0][i]=fq_quantum_liq[0][ch_liq]
                #     fq_quantum_liq[1][i]=fq_quantum_liq[1][ch_liq]
                #     fq_quantum_liq[2][i]=fq_quantum_liq[2][ch_liq]
            fq_quantum_sol=np.copy(fq_quantum)
            fq_quantum_sol=fq_quantum_sol[:,ch_liq:]
            liq_bcl=fq_quantum_liq[:,0]
            liq_bcr=fq_quantum_liq[:,-1]            
            sol_bcl=fq_quantum_sol[:,0]
            sol_bcr=fq_quantum_sol[:,-1]

            
            #rewrite so sol is shifted over, prevents overflow to the left letting heat escape
    #        print(ch_liq)
    
        
        fq_quantum_liq, qc2,ex_term,ex_c, ex_raw_l = oneTimeStep_quantumStreaming_liq(fq_quantum_liq, int(np.size(fq_quantum_liq)/3), numberOfShots, Aer.get_backend(backend), t, maxT,1,liq_bcl,liq_bcr)
        rhoq_quantum_liq = fq_quantum_liq[0] + fq_quantum_liq[1] + fq_quantum_liq[2]
        ex+=ex_term
        fq_quantum_sol, qc2,ex_term_sol,ex_c2, ex_raw_s = oneTimeStep_quantumStreaming(fq_quantum_sol, int(np.size(fq_quantum_sol)/3), numberOfShots, Aer.get_backend(backend), t, maxT,tm,sol_bcl,sol_bcr)
        rhoq_quantum_sol = fq_quantum_sol[0] + fq_quantum_sol[1] + fq_quantum_sol[2]
        ex+=ex_term_sol
        runsum_q+=ex_term
        runsum_q+=ex_term_sol
        runsum+=ex_c
        runsum+=ex_c2
        ex_plus+=ex_term
        ex_minus-=ex_term_sol
        
        print(ex_term)
        print(ex_c)
        print(ex_term_sol)
        print(ex_c2)
        # if np.mod(t,rate)==0 or t<8:
        #     for i in range(M-ch_liq):
        #         if i<ch_liq:
        #             fq_quantum[0][i]=fq_quantum_liq[0][i]
        #             fq_quantum[1][i]=fq_quantum_liq[1][i]
        #             fq_quantum[2][i]=fq_quantum_liq[2][i]
        #         elif i>ch_liq:
        #             fq_quantum[0][i]=fq_quantum_sol[0][i-ch_liq]
        #             fq_quantum[1][i]=fq_quantum_sol[1][i-ch_liq]
        #             fq_quantum[2][i]=fq_quantum_sol[2][i-ch_liq]  
        #         else:
        #             ex_heat=0
        #             for j in range(i,M-1):
        #                 ex_heat+=(fq_quantum_liq[0][j]+fq_quantum_liq[1][j]+fq_quantum_liq[2][j]-tm)
        #             fq_quantum[0][i]+=ex_heat*fq_quantum[0][i]/rhoq_quantum[i]
        #             fq_quantum[1][i]+=ex_heat*fq_quantum[0][i]/rhoq_quantum[i]
        #             fq_quantum[2][i]+=ex_heat*fq_quantum[0][i]/rhoq_quantum[i]
        if np.mod(t,min(rate,short))==0 or t<2*rate:  
            #print(runsum)
            #print(runsum_q)
            if ch_liq!=0:
                for i in range(M):
                    if i<ch_liq:
                        fq_quantum[0][i]=fq_quantum_liq[0][i]
                        fq_quantum[1][i]=fq_quantum_liq[1][i]
                        fq_quantum[2][i]=fq_quantum_liq[2][i]
                    elif i>ch_liq:
                        fq_quantum[0][i]=fq_quantum_sol[0][i-ch_liq]
                        fq_quantum[1][i]=fq_quantum_sol[1][i-ch_liq]
                        fq_quantum[2][i]=fq_quantum_sol[2][i-ch_liq]
                    else:
                        fq_quantum[0][i]=fq_quantum_sol[0][i-ch_liq]
                        fq_quantum[1][i]=fq_quantum_sol[1][i-ch_liq]
                        fq_quantum[2][i]=fq_quantum_liq[2][i]
                    #     #fq_quantum[0][i]=(fq_quantum_sol[0][i-ch_liq]*(q_state[i])+fq_quantum_sol[2][i-ch_liq]*(1-q_state[i]))
                    #     #fq_quantum[1][i]=(fq_quantum_liq[1][i]*(q_state[i])+fq_quantum_sol[1][i-ch_liq]*(1-q_state[i]))
                    #     #fq_quantum[2][i]=(fq_quantum_liq[0][i]*(q_state[i])+fq_quantum_liq[2][i]*(1-q_state[i]))
                    #     fq_quantum[0][i]=fq_quantum_sol[0][i-ch_liq]*(q_state[i])+fq_quantum_liq[0][i]*(1-q_state[i])
                    #     fq_quantum[1][i]=fq_quantum_sol[1][i-ch_liq]*q_state[i]+fq_quantum_liq[1][i]*(1-q_state[i])
                    #     fq_quantum[2][i]=fq_quantum_sol[2][i-ch_liq]*q_state[i]+fq_quantum_liq[2][i]*(1-q_state[i])

            rhoq_quantum = fq_quantum[0] + fq_quantum[1] + fq_quantum[2]
    full_liq=0
    ch_liq=0
    ch_sol=0
    full_sol=0
    for i in range(M):
        if rhoq_quantum[i]>=tm:
            if q_state[i]<1:
                # rho_liq=fq_quantum_liq[0][i]+fq_quantum_liq[1][i]+fq_quantum_liq[2][i]
                # rho_sol=fq_quantum_sol[0][i-ch_liq]+fq_quantum_sol[1][i-ch_liq]+fq_quantum_sol[2][i-ch_liq]
                
                # q_state[i]+=((rho_liq)*(1-q_state[i])+(rho_sol*q_state[i])-tm)*cp/la
                if not any(0 < x < 1 for x in q_state[1:]):
                    q_state[i]+=(rhoq_quantum[i]-tm)*cp/la
                q_state[i]+=(ex)*cp/la
                fq_quantum[0][i]*=tm/rhoq_quantum[i]
                fq_quantum[1][i]*=tm/rhoq_quantum[i]
                fq_quantum[2][i]*=tm/rhoq_quantum[i]
            if q_state[i]>=1:
                fq_quantum[0][i]+=fq_quantum[0][i]/rhoq_quantum[i]*(q_state[i]-1)*la/cp
                fq_quantum[1][i]+=fq_quantum[1][i]/rhoq_quantum[i]*(q_state[i]-1)*la/cp
                fq_quantum[2][i]+=fq_quantum[2][i]/rhoq_quantum[i]*(q_state[i]-1)*la/cp
                q_state[i]=1
        if q_state[i]==1:
            full_liq=i
            ch_liq=full_liq+1
            # elif q_state[i]<1 and q_state[i]>0:
            #     ch_liq=i
            # elif q_state[i]==0 and q_state[i-1]!=0:
            #     full_sol=i
            #     if q_state[i-1]!=1:
            #         ch_sol=i-1
    rhoq_quantum = fq_quantum[0] + fq_quantum[1] + fq_quantum[2]
    liq_frac=q_state[ch_liq]
    ex=0
    if t>5:
        if ch_liq!=ch_old:
            q_series=[]
            ch_old=ch_liq
            print(fq_quantum_liq)
            print(fq_quantum_sol)
        q_series.append([t, q_state[ch_liq]])
    if short<rate and count>0:
        short=1
        count-=1
        print(t)
    elif count==0:
        short=rate+1
    if t>5 and len(q_series)>=5 and short==rate+1:
        t_val = [item[0] for item in q_series]  
        y_val = [item[1] for item in q_series]  
        slope, intercept, _, _, _ = linregress(t_val, y_val)
        t_shift = (tb - intercept) / slope
        if t+rate>t_shift-1:
            short=1#np.floor(t+rate-t_shift)
            count=np.floor(10+t_shift-t)
            print(short)
            print(count)
        else:
            short=rate+1
    print(ch_liq)
    ex_minus=0
    ex_plus=0
    # fq_quantum_tot_skip[:,:,t]=fq_quantum
    # q_state_tot_skip[:,t]=q_state
    # ex_tot_skip[t]=ex
    # fq_quantum_tot[:,:,t]=fq_quantum
    # q_state_tot[:,t]=q_state
    # ex_tot[t]=ex
    # fq_quantum, qc2 = oneTimeStep_quantumStreaming(fq_quantum, M, numberOfShots, Aer.get_backend(backend), t, maxT)
    # rhoq_quantum = fq_quantum[0] + fq_quantum[1] + fq_quantum[2]
    #classical implementation
    fClassical,state_cc = classicalOneTimeStep(fClassical, state_cc, M,t,rate)
    rhoClassical = fClassical[0] + fClassical[1] + fClassical[2]
    
    all_fq_liq.append(fq_quantum_liq)
    all_fq_sol.append(fq_quantum_sol)
    all_q_state.append(np.copy(q_state))
    left_boundary_values_c.append([fClassical[0][0], fClassical[1][0], fClassical[2][0]])
    mid_boundary_values_c.append([fClassical[0][1], fClassical[1][1], fClassical[2][1]])
    right_boundary_values_c.append([fClassical[0][2], fClassical[1][2], fClassical[2][2]])
    
    left_boundary_values.append([fq_quantum[0][0], fq_quantum[1][0], fq_quantum[2][0]])
    mid_boundary_values.append([fq_quantum[0][1], fq_quantum[1][1], fq_quantum[2][1]])
    right_boundary_values.append([fq_quantum[0][2], fq_quantum[1][2], fq_quantum[2][2]])
    
    qq_state.append(q_state[ch_liq])
    cc_state.append(state_cc[ch_liq])

    
    loc=2*gam*np.sqrt(alpha*t*dt)
    key_num=int((M+1)/2)
    x_liq=np.zeros(int((M+1)/2))
    x_sol=np.zeros(key_num)
    x_base_liq=np.linspace(0,loc,int((M+1)/2))
    x_base_sol=np.linspace(loc,L,key_num)
    x_base=np.concatenate((x_base_liq,x_base_sol[1:]))
    for i in range(int((M+1)/2)):
        x_liq[i]=tb-(tb-tm)*(special.erf((i)*loc/((M+1)/2-1)/(2*np.sqrt(alpha*t*dt))))/special.erf(gam)
    for i in range(key_num):
        #x_sol[i] = (tm)*special.erfc(x_base_sol[i]/(2*np.sqrt(alpha*(t*dt)))) /special.erfc(gam)
        x_sol[i]=tm + (to - tm) * (x_base_sol[i]) / L + np.sum(
            (2 / (n_terms * np.pi)) * (to - tm) * np.sin(n_terms * np.pi * (x_base_sol[i]) / L) *
            np.exp(-alpha * (n_terms * np.pi / L)**2 * t)
        )
    #rho_quantum
    # Add the steady-state temperature distribution
    x_tot=np.concatenate((x_liq,x_sol[1:]))
    loc_mod.append(np.mod(loc,1))
    if np.mod(t,rate)==0:
        # Store the temperature at the chosen points
        boun_q=ch_liq-0.5+q_state[ch_liq]
        #if q_state[ch_liq]==0:
        #    boun_q+=0.5
        boun_c=ch_liq-0.5+state_cc[ch_liq]
        
        if q_state[ch_liq]!=0:
            qq_interp = np.concatenate([
            np.linspace(0, boun_q, ch_liq + 1),
            np.linspace(boun_q + (M - ch_liq - 1) / (M - ch_liq),boun_q + (M - ch_liq - 1) / (M - ch_liq)+M-ch_liq-2, M - ch_liq - 1)
        ])  
        else:
            qq_interp=np.linspace(0,M-1,M)
        if state_cc[ch_liq]!=0:    
            cc_interp = np.concatenate([
                np.linspace(0, boun_c, ch_liq + 1),
                np.linspace(boun_c + (M - ch_liq - 1) / (M - ch_liq),boun_c + (M - ch_liq - 1) / (M - ch_liq)+M-ch_liq-2, M - ch_liq - 1)
            ])   
        
        else:
            cc_interp=np.linspace(0,M-1,M)
        
        temps_at_qq = heat_transfer_interp(t, dt, alpha, gam, L, M, tb, tm, to, n_terms, qq_interp)

        try:
            q_tot_interp = [interpolate_x_tot(qq_interp, rhoq_quantum, point) for point in range(M)]
            c_tot_interp = [interpolate_x_tot(cc_interp, rhoClassical, point) for point in range(M)]
            MSE=math.sqrt(mean_squared_error(qq_interp,cc_interp))
            errs.append(MSE)
            
            plt.plot(rate*np.linspace(0,len(errs[4:])-1,len(errs[4:])) ,errs[4:])
            plt.ylabel('Error', fontsize='large')
            plt.xlabel('Timesteps', fontsize='large')
            plt.savefig('error_1.pdf', format='pdf', bbox_inches='tight', dpi=300)
            #plt.title('Error between quantum and analytical solutions')
            plt.show()
        except ValueError as e:
            e
        if t==0:
            temperature_at_point_liq.append(0)
            temperature_at_point_sol.append(0)
            temperature_at_point_liq_c.append(0)
            temperature_at_point_sol_c.append(0)
        else:           
            temperature_at_point_liq.append(interpolate_x_tot(qq_interp,rhoq_quantum,point_liq))
            temperature_at_point_sol.append(interpolate_x_tot(qq_interp,rhoq_quantum,point_sol))
            temperature_at_point_liq_c.append(interpolate_x_tot(cc_interp,rhoClassical,point_liq))
            temperature_at_point_sol_c.append(interpolate_x_tot(cc_interp,rhoClassical,point_sol))
        if t==0:
            temperature_at_point_liq_a.append(0)
            temperature_at_point_sol_a.append(0)
        else:
            temperature_at_point_liq_a.append(interpolate_x_tot(x_base,x_tot,point_liq))
            temperature_at_point_sol_a.append(interpolate_x_tot(x_base,x_tot,point_sol))
        time_series.append(t * dt)
        
        #if t==0:
        #    temperature_at_point_liq_a.append(0)
        #    temperature_at_point_sol_a.append(0)
        # x_base=np.linspace(0,L,M)
        # x_tot=np.zeros(M)
        # for i in range(M):
        #     if i<loc:
        #         x_tot[i]=tb-(tb-tm)*(special.erf(i/(2*np.sqrt(alpha*t*dt))))/special.erf(gam)
        #     else:
        #         x_tot[i]=to+(tm-to)*(special.erfc(i/(2*np.sqrt(alpha*t*dt))))/special.erfc(gam)

        # plt.plot(x, rhoq_classical, markersize = 6, markerfacecolor='red')
        
        
        
        
        plt.plot(qq_interp, rhoq_quantum,'s', markersize = 6, markerfacecolor='none')
        plt.plot(cc_interp, rhoClassical, 'k')
        plt.plot(x_base,x_tot)
        plt.xlabel('Lattice site', fontsize='large')
        plt.ylabel('Temperature', fontsize='large')
        plt.legend(['Quantum', 'Classical', 'Analytic'], fontsize='large')
        #plt.show()
        plt.savefig('LvT_1.pdf', format='pdf', bbox_inches='tight', dpi=300)
        plt.show()
        
        qq_interp_all.append(qq_interp)
        cc_interp_all.append(cc_interp)
        rho_q_all.append(rhoq_quantum)
        rho_c_all.append(rhoClassical)
        x_base_all.append(x_base)
        x_tot_all.append(x_tot)
        
        
        # Define colors for each type
        quantum_color = 'blue'
        classical_color = 'red'
        analytical_color = 'green'
        
        # Plot with same colors but different line styles
        plt.plot(time_series, temperature_at_point_liq, color=quantum_color, linestyle='-', label='Quantum')
        plt.plot(time_series, temperature_at_point_sol, color=quantum_color, linestyle='--')
        
        plt.plot(time_series, temperature_at_point_liq_c, color=classical_color, linestyle='-', label='Classical')
        plt.plot(time_series, temperature_at_point_sol_c, color=classical_color, linestyle='--')
        
        plt.plot(time_series, temperature_at_point_liq_a, color=analytical_color, linestyle='-', label='Analytical')
        plt.plot(time_series, temperature_at_point_sol_a, color=analytical_color, linestyle='--')
        
        plt.xlabel('Time', fontsize='large')
        plt.ylabel('Temperature',fontsize='large')
        plt.legend()
        plt.savefig('TvT_1.pdf', format='pdf', bbox_inches='tight', dpi=300)
        plt.show()
        
        
        # In your simulation loop:
        for i in range(num_nodes_to_track-1):
            qq_states[i].append(q_state[i+1])
            cc_states[i].append(state_cc[i+1])
        
        # Plotting
        colors = ['blue', 'red', 'green', 'purple', 'orange']
        plt.figure(figsize=(10, 6))
        for i in range(num_nodes_to_track-1):
            plt.plot(qq_states[i], label=f'Quantum Node {i}', 
                     color=colors[i], linestyle='-')
            plt.plot(cc_states[i], label=f'Classical Node {i}', 
                     color=colors[i], linestyle='--')

        plt.xlabel("Timestep")
        plt.ylabel('Liquid fraction')
        plt.legend()
        #plt.grid(True)
        plt.show()
        # plt.plot(cc_state)
        # plt.plot(qq_state)
        # plt.xlabel("Timestep")
        # plt.ylabel('Liquid fraction')
        # plt.legend(['Quantum','Classical'])
        # plt.show()
        x_val=[pair[0]for pair in lin_check]
        y_val=[pair[1]**2*np.sign(pair[1]) for pair in lin_check]
        #plt.scatter(x_val,y_val)
        #plt.plot(np.linspace(0,0.5,100),np.linspace(0,0.5,100))
        plt.show()
        if (x_val!=[]): 
            corr=linregress(x_val,y_val)
    # if t==150:
    #     np.savetxt("fq_tot_rate1.txt",fq_quantum_tot.reshape(fq_quantum_tot.shape[0],-1))
    #     np.savetxt("q_state_rate1.txt",q_state_tot)
    #     np.savetxt("ex_rate1.txt",ex_tot)

print('Done')
def process_and_save(data_list, filename):
    # Find the maximum number of columns
    max_cols = max(arr.shape[1] for arr in data_list)

    # Create a list to hold our processed data
    processed_data = []

    for arr in data_list:
        # Pad each array to have the same number of columns
        padded = np.pad(arr, ((0, 0), (0, max_cols - arr.shape[1])), mode='constant', constant_values=np.nan)

        # Flatten the 2D array into a 1D array
        flattened = padded.flatten()

        processed_data.append(flattened)

    # Create column names
    columns = [f'row{i+1}_col{j+1}' for i in range(3) for j in range(max_cols)]

    # Create DataFrame
    df = pd.DataFrame(processed_data, columns=columns)

    # Save to CSV
    df.to_csv(filename, index=False)

    print(f"Saved {filename} with shape {df.shape}")

# Process and save both lists
process_and_save(all_fq_liq, "fq_liq.csv")
process_and_save(all_fq_sol, "fq_sol.csv")
all_q_df=pd.DataFrame(all_q_state)
all_q_df.to_csv("all_q_state.csv")


MSE = mean_squared_error(rhoClassical, rhoq_classical)
RMSE = math.sqrt(MSE)
print("RMSE of quantum solution with classical streaming:\n", RMSE)

MSE = mean_squared_error(rhoClassical, rhoq_quantum)
RMSE = math.sqrt(MSE)
print("RMSE of quantum solution with quantum streaming:\n", RMSE)
np.savetxt("fq_quantum.txt",fq_quantum)
np.savetxt("q_state.txt",q_state)
print(ch_liq)

time_steps = range(len(left_boundary_values))
time_steps=time_steps[3:]
# Convert lists to arrays for easier plotting
left_boundary_values_c = np.array(left_boundary_values_c[3:])
mid_boundary_values_c = np.array(mid_boundary_values_c[3:])

left_boundary_values = np.array(left_boundary_values[3:])
mid_boundary_values = np.array(mid_boundary_values[3:])

# Plotting the three components of left_boundary_values over time
plt.figure(figsize=(10, 5))
#classical
plt.plot(time_steps, left_boundary_values_c[:, 2], label='f[0][0]', marker='o')
plt.plot(time_steps, left_boundary_values_c[:, 1], label='f[1][0]', marker='o')
plt.plot(time_steps, left_boundary_values_c[:, 0], label='f[2][0]', marker='o')
#quantum
plt.plot(time_steps, left_boundary_values[:, 0], label='f[0][0]', marker='o')
plt.plot(time_steps, left_boundary_values[:, 1], label='f[1][0]', marker='o')
plt.plot(time_steps, left_boundary_values[:, 2], label='f[2][0]', marker='o')
plt.xlabel('Time Steps')
plt.ylabel('Left Boundary Values')
#plt.title('Left Boundary Values Over Time')
plt.legend()
plt.grid(True)
plt.show()

# Plotting the three components of mid_boundary_values over time
plt.figure(figsize=(10, 5))
#classical
plt.plot(time_steps, mid_boundary_values_c[:, 2], label='f[0][1]', marker='o')
plt.plot(time_steps, mid_boundary_values_c[:, 1], label='f[1][1]', marker='o')
plt.plot(time_steps, mid_boundary_values_c[:, 0], label='f[2][1]', marker='o')
#quantum
plt.plot(time_steps, mid_boundary_values[:, 0], label='f[0][1]', marker='o')
plt.plot(time_steps, mid_boundary_values[:, 1], label='f[1][1]', marker='o')
plt.plot(time_steps, mid_boundary_values[:, 2], label='f[2][1]', marker='o')
plt.xlabel('Time Steps')
plt.ylabel('Interface Boundary Values')
#plt.title('Mid Boundary Values Over Time')
plt.legend()
plt.grid(True)
plt.show()


np.save("left_boundary_c",left_boundary_values_c)
np.save("mid_boundary_c",mid_boundary_values_c)
np.save("left_boundary",left_boundary_values)
np.save("mid_boundary",mid_boundary_values)
np.save("time_series", time_series)
np.save("temp_liq",temperature_at_point_liq)
np.save("temp_sol",temperature_at_point_sol)
np.save("temp_liq_c",temperature_at_point_liq_c)
np.save("temp_sol_c",temperature_at_point_sol_c)
np.save("temp_liq_a",temperature_at_point_liq_a)
np.save("temp_sol_a",temperature_at_point_sol_a)
np.save("q_interp",qq_interp_all)
np.save("c_interp",cc_interp_all)
np.save("a_interp", x_tot_all)
np.save("rho_q", rho_q_all)
np.save("rho_c", rho_c_all)
np.save("rho_a",x_base_all)
