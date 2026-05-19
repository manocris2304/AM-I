import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
from scipy.linalg import eigh


np.set_printoptions(linewidth=np.inf)
#======================================================
# 1-Função para ler matriz do Abaqus

def read_abaqus_matrix(file_path):

    """
    Lê matriz exportada do Abaqus no formato:
    linha coluna valor
    """

    rows = []
    cols = []
    values = []

    with open(file_path, 'r') as f:
        for line in f:

            # Remove espaços extras
            line = line.strip()

            # Ignora linhas vazias
            if not line:
                continue

            # Divide a linha
            parts = line.split()

            # Garante que existem 3 colunas
            if len(parts) != 3:
                continue

            row = int(parts[0]) - 1   # Abaqus começa em 1
            col = int(parts[1]) - 1
            value = float(parts[2])

            rows.append(row)
            cols.append(col)
            values.append(value)

    # Descobre tamanho da matriz
    n_rows = max(rows) + 1
    n_cols = max(cols) + 1

    # Cria matriz esparsa
    matrix = coo_matrix((values, (rows, cols)),
                   shape=(n_rows, n_cols))

    # Converte para CSR (mais eficiente)
    matrix = matrix.tocsr()

    return matrix

# Matrizes globais , variaveis do modelo e da redução
file_pathK = "5_STIF1_coord.mtx"
Kfull = read_abaqus_matrix(file_pathK)
Kfull = Kfull.toarray() #Converte para matriz densa 
file_pathM = "5_MASS1_coord.mtx"
Mfull = read_abaqus_matrix(file_pathM)
Mfull = Mfull.toarray() #Converte para matriz densa
num_linhas = Kfull.shape[0]
num_colunas = Kfull.shape[1]
if num_linhas != num_colunas:
    print("A matriz de rigidez não é quadrada. Verifique os arquivos de entrada.")
    exit()
print("\nNúmero total de graus de liberdade:", num_linhas)
dof_per_node = 6
n_modos = dof_per_node + 10
n_nodes = int(num_linhas/dof_per_node)
print("\nNúmero total de nós:", int(n_nodes))
#======================================================
# 2-Cálculo dos 10 primeiros modos da matriz
def calcular_modos(K, M, n_modos):
    autovalores, autovetores = eigsh(K,k=n_modos,M=M,sigma=1e-6,which='LM')

    #=======================================================

    # lambda = omega²
    omega = np.sqrt(np.abs(autovalores))

    # Hz
    frequencias = omega / (2 * np.pi)
    return frequencias
#=======================================================
# Modos Matriz completa
frequencias = calcular_modos(Kfull, Mfull, n_modos)
print("\n===== FREQUÊNCIAS NATURAIS =====\n")
for i, freq in enumerate(frequencias):

    print(f"Modo {i+1}: {freq:.6f} Hz")
#=======================================================
# 3-Seleção dos nós/graus de liberdade mestres para redução
def selecionar_mestres(num_linhas, dof_per_node, p_reducao):

    n_nodes = int(num_linhas/dof_per_node)
    n_mestres = int(n_nodes*(1-p_reducao/100))
    passo_reducao = int((n_nodes-1)/(n_mestres-1)) #Passo entre os nós mestres
    master_nodes = np.linspace(0,n_nodes-1,n_mestres,dtype=int)
    master_dofs = []

    for node in master_nodes:

        for dof in range(dof_per_node):

            master_dofs.append(node * dof_per_node + dof)

    master_dofs = np.array(master_dofs)
    return master_dofs

#======================================================
# 4- Guyan
def guyan (Kg, Mg, n_dof, mestre):

    slave = np.delete(np.arange(n_dof), mestre) #Exclui os elementos mestres do vetor de elementos escravos
  
    K_mm = Kg[np.ix_(mestre, mestre)] #Cria matriz de rigidez apenas com elementos mestres (ativos)
    M_mm = Mg[np.ix_(mestre, mestre)]#Cria matriz de massa apenas com elementos mestres (ativos)

    K_ss = Kg[np.ix_(slave, slave)] #Cria matriz de rigidez apenas com elementos escravos (deletados)
    M_ss = Mg[np.ix_(slave, slave)] #Cria matriz de massa apenas com elementos mestres (deletados)

    K_ms = Kg[np.ix_(mestre, slave)] #Cria matriz de rigidez mestre-escravo
    K_sm= np.transpose(K_ms) #Cria matriz de rigidez escravo-mestre
    M_ms = Mg[np.ix_(mestre, slave)] #Cria matriz de massa mestre-escravo
    M_sm = np.transpose(M_ms) #Cria matriz de massa escravo-mestre

    K_guyan = np.block([[K_mm, K_ms], [K_sm, K_ss]]) #Matriz de rigidez global organizada em blocos
    M_guyan = np.block([[M_mm, M_ms], [M_sm, M_ss]]) #Matriz de massa global organizada em blocos

    ts= -np.linalg.inv(K_ss)@K_sm #Matriz de transformação dos escravos em função dos mestres
    I = np.eye(ts.shape[1]) #Matriz identidade para os graus de liberdade mestres
    Ts = np.vstack((I, ts)) #Matriz de transformação completa, combinando mestres e escravos

    Krg= np.transpose(Ts)@K_guyan@Ts #Matriz de rigidez reduzida por Guyan
    Mrg = np.transpose(Ts)@M_guyan@Ts #Matriz de massa reduzida por Guyan

    return Krg, Mrg

#====================================================== 
# 5- SEREP
def SEREP (Kg, Mg, n_dof, mestre, n_modos):
    
    slave = np.delete(np.arange(n_dof), mestre) #Exclui os elementos mestres do vetor de elementos escravos
   
    K_mm = Kg[np.ix_(mestre, mestre)] #Cria matriz de rigidez apenas com elementos mestres (ativos)
    M_mm = Mg[np.ix_(mestre, mestre)]#Cria matriz de massa apenas com elementos mestres (ativos)

    K_ss = Kg[np.ix_(slave, slave)] #Cria matriz de rigidez apenas com elementos escravos (deletados)
    M_ss = Mg[np.ix_(slave, slave)] #Cria matriz de massa apenas com elementos mestres (deletados) crispim broxa
    
    K_ms = Kg[np.ix_(mestre, slave)]
    K_sm= np.transpose(K_ms)
    M_ms = Mg[np.ix_(mestre, slave)]
    M_sm = np.transpose(M_ms)

    K_serep = np.block([[K_mm, K_ms], [K_sm, K_ss]])
    M_serep = np.block([[M_mm, M_ms], [M_sm, M_ss]])

    autovalor, PHI = eigh(Kg,Mg)
    PHI = PHI [:, :n_modos]
    PHI_m = PHI[mestre, :]
    PHI_s = PHI[slave, :]
    
    tu = PHI_s@np.linalg.pinv(PHI_m)
    I=np.eye(tu.shape[1])
    Tu=np.vstack((I,tu))

    Kr=np.transpose(Tu)@K_serep@Tu
    Mr=np.transpose(Tu)@M_serep@Tu

    return Kr, Mr

#======================================================
# Calculo de convergência de Guyan e SEREP
convergencia_Guyan = []
convergencia_SEREP = []
for i in range(5, 105, 10):
    p_reducao = i
    j=i/10-1
    master_dofs = selecionar_mestres(num_linhas, dof_per_node, p_reducao) 
    Kr_guyan, Mr_guyan = guyan(Kfull, Mfull, num_linhas, master_dofs)
    frequencias_guyan = calcular_modos(Kr_guyan, Mr_guyan, n_modos)
    num_linhas_guyan = Kr_guyan.shape[0]
    coluna_guyan = np.concatenate(([i],[num_linhas_guyan],frequencias_guyan))
    convergencia_Guyan.append(coluna_guyan)

    Kr_SEREP, Mr_SEREP = SEREP(Kfull, Mfull, num_linhas, master_dofs,n_modos)
    num_linhas_SEREP = Kr_SEREP.shape[0]
    frequencias_SEREP = calcular_modos(Kr_SEREP, Mr_SEREP, n_modos)
    coluna_SEREP = np.concatenate(([i],[num_linhas_SEREP],frequencias_SEREP))
    convergencia_SEREP.append(coluna_SEREP)

convergencia_Guyan = np.array(convergencia_Guyan).T
convergencia_SEREP = np.array(convergencia_SEREP).T


print("\nConvergência Guyan:\n")
print(convergencia_Guyan)

print("\nConvergência SEREP:\n")
print(convergencia_SEREP)