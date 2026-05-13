import numpy as np
from scipy.linalg import eigh

# ============================================================
# LEITURA MATRIZ ABAQUS
# ============================================================

def ler_matriz_abaqus(arquivo_mtx, ndof_por_no=6):

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

                entradas.append(
                    (
                        node_i,
                        dof_i,
                        node_j,
                        dof_j,
                        value
                    )
                )

                max_node = max(
                    max_node,
                    node_i,
                    node_j
                )

            except:
                continue

    n_dof_total = max_node * ndof_por_no

    K = np.zeros(
        (
            n_dof_total,
            n_dof_total
        )
    )

    # ========================================================
    # MONTAGEM
    # ========================================================

    for node_i, dof_i, node_j, dof_j, value in entradas:

        row = (
            (node_i - 1) * ndof_por_no
            + (dof_i - 1)
        )

        col = (
            (node_j - 1) * ndof_por_no
            + (dof_j - 1)
        )

        K[row, col] += value

    # ========================================================
    # SIMETRIZAÇÃO
    # ========================================================

    K = 0.5 * (K + K.T)

    return K

# ============================================================
# CONVERSÃO NÓS -> GDL
# ============================================================

def converter_nos_para_gdl(
        master_nodes,
        ndof_por_no=6
):

    mestre = []

    for node in master_nodes:

        # apenas UX UY UZ

        for dof in range(3):

            gdof = (
                (node - 1) * ndof_por_no
                + dof
                + 1
            )

            mestre.append(gdof)

    return mestre

# ============================================================
# ESTABILIZAÇÃO
# ============================================================

def estabilizar_matriz(
        A,
        eps=1e-8
):

    A = 0.5 * (A + A.T)

    A += eps * np.eye(A.shape[0])

    return A

# ============================================================
# MODAL
# ============================================================

def modal(K, M):

    K = estabilizar_matriz(K)
    M = estabilizar_matriz(M)

    # ========================================================
    # REMOVE MASSA SINGULAR
    # ========================================================

    valM, vecM = np.linalg.eigh(M)

    tol = 1e-10

    positivos = valM > tol

    valM = valM[positivos]

    vecM = vecM[:, positivos]

    M = (
        vecM
        @ np.diag(valM)
        @ vecM.T
    )

    M = estabilizar_matriz(M)

    # ========================================================
    # AUTOVALORES
    # ========================================================

    autovalores, autovetores = eigh(K, M)

    autovalores = np.real(autovalores)

    autovalores[autovalores < 0] = 0

    return autovalores

# ============================================================
# GUYAN
# ============================================================

def guyan(
        Kg,
        Mg,
        n_dof,
        mestre
):

    mestre = np.array(mestre) - 1

    gdl = np.arange(n_dof)

    slave = np.setdiff1d(
        gdl,
        mestre
    )

    # ========================================================
    # PARTIÇÕES
    # ========================================================

    K_mm = Kg[np.ix_(mestre, mestre)]

    M_mm = Mg[np.ix_(mestre, mestre)]

    K_ss = Kg[np.ix_(slave, slave)]

    M_ss = Mg[np.ix_(slave, slave)]

    K_ms = Kg[np.ix_(mestre, slave)]

    K_sm = K_ms.T

    M_ms = Mg[np.ix_(mestre, slave)]

    M_sm = M_ms.T

    # ========================================================
    # ESTABILIZAÇÃO
    # ========================================================

    K_ss = estabilizar_matriz(K_ss)

    # ========================================================
    # TRANSFORMAÇÃO
    # ========================================================

    ts = -np.linalg.solve(K_ss, K_sm)

    I = np.eye(len(mestre))

    Ts = np.vstack((I, ts))

    # ========================================================
    # MATRIZES
    # ========================================================

    K_guyan = np.block([
        [K_mm, K_ms],
        [K_sm, K_ss]
    ])

    M_guyan = np.block([
        [M_mm, M_ms],
        [M_sm, M_ss]
    ])

    Kr = Ts.T @ K_guyan @ Ts

    Mr = Ts.T @ M_guyan @ Ts

    Kr = estabilizar_matriz(Kr)

    Mr = estabilizar_matriz(Mr)

    return Kr, Mr

# ============================================================
# SEREP
# ============================================================

def SEREP(
        Kg,
        Mg,
        n_dof,
        mestre
):

    mestre = np.array(mestre) - 1

    gdl = np.arange(n_dof)

    slave = np.setdiff1d(
        gdl,
        mestre
    )

    # ========================================================
    # MODAL COMPLETO
    # ========================================================

    autovalor, PHI = eigh(
        estabilizar_matriz(Kg),
        estabilizar_matriz(Mg)
    )

    validos = autovalor > 1e-8

    autovalor = autovalor[validos]

    PHI = PHI[:, validos]

    # ========================================================
    # MODOS
    # ========================================================

    n_modos = min(
        6,
        PHI.shape[1],
        len(mestre)
    )

    PHI = PHI[:, :n_modos]

    PHI_m = PHI[mestre, :]

    PHI_s = PHI[slave, :]

    # ========================================================
    # TRANSFORMAÇÃO
    # ========================================================

    ts = PHI_s @ np.linalg.pinv(
        PHI_m,
        rcond=1e-8
    )

    I = np.eye(ts.shape[1])

    Ts = np.vstack((I, ts))

    # ========================================================
    # PARTIÇÕES
    # ========================================================

    K_mm = Kg[np.ix_(mestre, mestre)]

    M_mm = Mg[np.ix_(mestre, mestre)]

    K_ss = Kg[np.ix_(slave, slave)]

    M_ss = Mg[np.ix_(slave, slave)]

    K_ms = Kg[np.ix_(mestre, slave)]

    K_sm = K_ms.T

    M_ms = Mg[np.ix_(mestre, slave)]

    M_sm = M_ms.T

    K_serep = np.block([
        [K_mm, K_ms],
        [K_sm, K_ss]
    ])

    M_serep = np.block([
        [M_mm, M_ms],
        [M_sm, M_ss]
    ])

    Kr = Ts.T @ K_serep @ Ts

    Mr = Ts.T @ M_serep @ Ts

    # ========================================================
    # REMOVE MASSA SINGULAR
    # ========================================================

    valM, vecM = np.linalg.eigh(
        estabilizar_matriz(Mr)
    )

    positivos = valM > 1e-10

    valM = valM[positivos]

    vecM = vecM[:, positivos]

    Mr = (
        vecM
        @ np.diag(valM)
        @ vecM.T
    )

    Kr = (
        vecM.T
        @ Kr
        @ vecM
    )

    Kr = estabilizar_matriz(Kr)

    Mr = estabilizar_matriz(Mr)

    return Kr, Mr

# ============================================================
# CONFIGURAÇÃO
# ============================================================

ndof_por_no = 6

arquivo_K = '2_5_STIF1_label.mtx'

arquivo_M = '2_5_MASS1_label.mtx'

# ============================================================
# LEITURA
# ============================================================

print('\nLendo matrizes...')

Kg = ler_matriz_abaqus(
    arquivo_K,
    ndof_por_no
)

Mg = ler_matriz_abaqus(
    arquivo_M,
    ndof_por_no
)

print('Matrizes carregadas.')

# ============================================================
# REMOVE GDL SEM MASSA
# ============================================================

print('\nRemovendo GDLs sem massa...')

diagM = np.diag(Mg)

ativos = np.where(
    np.abs(diagM) > 1e-12
)[0]

Kg = Kg[np.ix_(ativos, ativos)]

Mg = Mg[np.ix_(ativos, ativos)]

print('GDLs ativos:', Kg.shape[0])

# ============================================================
# NÓS MESTRES
# ============================================================

n_nos = Kg.shape[0] // ndof_por_no

# pega de 5 em 5

master_nodes = list(
    range(
        1,
        n_nos + 1,
        5
    )
)

mestre = converter_nos_para_gdl(
    master_nodes,
    ndof_por_no
)

# ============================================================
# AJUSTE APÓS FILTRO
# ============================================================

mestre = np.array(mestre) - 1

mestre = np.intersect1d(
    mestre,
    ativos
)

mapa = {
    old: new
    for new, old
    in enumerate(ativos)
}

mestre = [
    mapa[g] + 1
    for g in mestre
]

print('\nNós mestres:')

print(master_nodes)

print('\nQuantidade GDLs mestres:')

print(len(mestre))

# ============================================================
# REDUÇÃO
# ============================================================

n_dof = Kg.shape[0]

print('\nExecutando Guyan...')

Kr_guyan, Mr_guyan = guyan(
    Kg,
    Mg,
    n_dof,
    mestre
)

print('Guyan concluído.')

print('\nExecutando SEREP...')

Kr_serep, Mr_serep = SEREP(
    Kg,
    Mg,
    n_dof,
    mestre
)

print('SEREP concluído.')

# ============================================================
# FREQUÊNCIAS
# ============================================================

print('\nCalculando frequências...')

fn_completo = np.sqrt(
    modal(Kg, Mg)
)

fn_guyan = np.sqrt(
    modal(
        Kr_guyan,
        Mr_guyan
    )
)

fn_serep = np.sqrt(
    modal(
        Kr_serep,
        Mr_serep
    )
)

print('\nFrequências modelo completo:\n')

print(fn_completo)

print('\nFrequências Guyan:\n')

print(fn_guyan)

print('\nFrequências SEREP:\n')

print(fn_serep)

# ============================================================
# SALVAR
# ============================================================

np.savetxt(
    'Kr_guyan.txt',
    Kr_guyan
)

np.savetxt(
    'Mr_guyan.txt',
    Mr_guyan
)

np.savetxt(
    'Kr_serep.txt',
    Kr_serep
)

np.savetxt(
    'Mr_serep.txt',
    Mr_serep
)

print('\nArquivos salvos com sucesso.')