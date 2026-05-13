import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh

def ler_matriz_abaqus(arquivo_mtx, ndof_por_no):

    entradas = []
    max_node = 0

    with open(arquivo_mtx, 'r') as f:

        for line in f:
            line = line.strip()

            if not line:
                continue

            parts = line.split(',')

            if len(parts) < 5:
                continue

            try:

                node_i = int(parts[0])
                dof_i = int(parts[1])

                node_j = int(parts[2])
                dof_j = int(parts[3])

                value = float(parts[4])

                entradas.append((node_i,dof_i,node_j,dof_j,value))
                max_node = max(max_node, node_i, node_j)
            except:
                continue
    n_dof_total = max_node * ndof_por_no

    K = np.zeros((n_dof_total, n_dof_total))

    for node_i, dof_i, node_j, dof_j, value in entradas:
        row = ((node_i - 1) * ndof_por_no + (dof_i - 1))
        col = ((node_j - 1) * ndof_por_no + (dof_j - 1))
        K[row, col] += value

    K = 0.5 * (K + K.T)

    return K

def converter_nos_para_gdl(master_nodes, ndof_por_no):
    mestre = []
    for node in master_nodes:
        for dof in range(ndof_por_no):

            gdof = (node - 1) * ndof_por_no + dof + 1
            mestre.append(gdof)

    return mestre

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

    n_modos = min(4, len(mestre))
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

ndof_por_no = 6

Kg = ler_matriz_abaqus('2_5_STIF1_label.mtx', ndof_por_no)
Mg =  ler_matriz_abaqus('2_5_MASS1_label.mtx', ndof_por_no)
n_dof = Kg.shape[0]
no_mestre = [1, 10, 25, 50, 100, 125, 150, 200]

mestre = converter_nos_para_gdl(no_mestre, ndof_por_no)

K_mestre = Kg [np.ix_(mestre, mestre)] #Cria matriz de rigidez apenas com elementos mestres (ativos)
M_mestre = Mg [np.ix_(mestre, mestre)]

Kr_guyan, Mr_guyan = guyan(Kg, Mg, n_dof, mestre)
Kr_serep, Mr_serep = SEREP(Kg, Mg, n_dof, mestre)

#Calculando as frequências naturais de acordo com cada método

fn_mestre = np.sqrt(modal(K_mestre, M_mestre))/(2*np.pi)
fn_guyan = np.sqrt(modal(Kr_guyan, Mr_guyan)) /(2*np.pi)
fn_serep = np.sqrt(modal(Kr_serep, Mr_serep)) /(2*np.pi)

print ('Frequencias naturais dos gdl mestres [Hz]:\n \n', fn_mestre)
print ('\nFrequencias naturais por Guyan [Hz]:\n \n', fn_guyan)
print ('\nFrequencias naturais por SEREP[Hz]:\n \n', fn_serep)







