import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh

def modal (K, M):
    autovalores, autovetores = eigh(K, M)
    autovalores = np.real(autovalores)
    return autovalores


def guyan (Kg, Mg, n_dof, mestre):
    mestre = np.array(mestre)-1 #O índice nessa linguagem de programação começa em 0
    gdl = np.arange(n_dof)  #Cria um vetor com todos os gdls
    slave = np.delete(gdl, mestre) #Exclui os elementos mestres do vetor de elementos escravos
   
    K_mm = Kg[np.ix_(mestre, mestre)] #Cria matriz de rigidez apenas com elementos mestres (ativos)
    M_mm = Mg[np.ix_(mestre, mestre)]#Cria matriz de massa apenas com elementos mestres (ativos)

    K_ss = Kg[np.ix_(slave, slave)] #Cria matriz de rigidez apenas com elementos escravos (deletados)
    M_ss = Mg[np.ix_(slave, slave)] #Cria matriz de massa apenas com elementos mestres (deletados)
    
    K_ms = Kg[np.ix_(mestre, slave)]
    K_sm= np.transpose(K_ms)
    M_ms = Mg[np.ix_(mestre, slave)]
    M_sm = np.transpose(M_ms)

    K_guyan = np.block([[K_mm, K_ms], [K_sm, K_ss]])
    M_guyan = np.block([[M_mm, M_ms], [M_sm, M_ss]])

    ts= -np.linalg.inv(K_ss)@K_sm
    I = np.eye(ts.shape[1])
    Ts = np.vstack((I, ts))

    Kr = np.transpose(Ts)@K_guyan@Ts
    Mr = np.transpose(Ts)@M_guyan@Ts

    return Kr, Mr

def SEREP (Kg, Mg, n_dof, mestre):
    mestre = np.array(mestre)-1 #O índice nessa linguagem de programação começa em 0
    gdl = np.arange(n_dof)  #Cria um vetor com todos os gdls
    slave = np.delete(gdl, mestre) #Exclui os elementos mestres do vetor de elementos escravos
   
    K_mm = Kg[np.ix_(mestre, mestre)] #Cria matriz de rigidez apenas com elementos mestres (ativos)
    M_mm = Mg[np.ix_(mestre, mestre)]#Cria matriz de massa apenas com elementos mestres (ativos)

    K_ss = Kg[np.ix_(slave, slave)] #Cria matriz de rigidez apenas com elementos escravos (deletados)
    M_ss = Mg[np.ix_(slave, slave)] #Cria matriz de massa apenas com elementos mestres (deletados)
    
    K_ms = Kg[np.ix_(mestre, slave)]
    K_sm= np.transpose(K_ms)
    M_ms = Mg[np.ix_(mestre, slave)]
    M_sm = np.transpose(M_ms)

    K_serep = np.block([[K_mm, K_ms], [K_sm, K_ss]])
    M_serep = np.block([[M_mm, M_ms], [M_sm, M_ss]])

    n_modos = len(mestre)
    autovalor, PHI = eigh(Kg, Mg)
    PHI = PHI [:, :n_modos]
    PHI_m = PHI[mestre, :]
    PHI_s = PHI[slave, :]
    
    tu = PHI_s@np.linalg.pinv(PHI_m)
    I=np.eye(tu.shape[1])
    Tu=np.vstack((I,tu))

    Kr=np.transpose(Tu)@K_serep@Tu
    Mr=np.transpose(Tu)@M_serep@Tu

    return Kr, Mr


Kg = np.array ([[10, -2, 0, 0, 0], [-2, 8, -1, 0, 0], [0, -1, 6, -1, 0], [0, 0, -1, 4, -1], [0, 0, 0, -1, 2]])
Mg = np.array ([[10, -2, 0, 0, 0], [-2, 8, -1, 0, 0], [0, -1, 6, -1, 0], [0, 0, -1, 4, -1], [0, 0, 0, -1, 2]])
n_dof = Kg.shape[0]
mestre = [1, 3]

Kr_guyan, Mr_guyan = guyan(Kg, Mg, n_dof, mestre)

Kr_serep, Mr_serep = SEREP(Kg, Mg, n_dof, mestre)

#Calculando as frequências naturais de acordo com cada método

fn_mestre = np.sqrt(modal(Kg, Mg))
fn_guyan = np.sqrt(modal(Kr_guyan, Mr_guyan))
fn_serep = np.sqrt(modal(Kr_serep, Mr_serep))

print ('Frequencias naturais dos gdl mestres:\n', fn_mestre)
print ('Frequencias naturais por Guyan:\n', fn_guyan)
print ('Frequencias naturais por SEREP:\n', fn_serep)


print (np.round(Kr_guyan, 4), '\n', np.round(Mr_guyan, 4))
print (np.round(Kr_serep, 4), '\n', np.round(Mr_serep, 4))





