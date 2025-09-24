import numpy as np
from scipy.sparse.linalg import eigsh

# ========= INITIAL VARIABLES ========== #
PH_CUT_Type = "True" # Local / Total for future
PH_MAX = 8 # Phonon occupation: from 0 to PH_MAX-1

EL_SITE_NUM = 2 # Number of electron sites: from 0 to EL_SITE_NUM-1

t = 1
omega = 1
g = 0.1
# ====================================== #

b_matrix = np.zeros((PH_MAX, PH_MAX)) # destruction operator, b
for i in range(1, PH_MAX):
    b_matrix[i-1, i] = np.sqrt(i) # <i-1|b|i> = sqrt(i)*<i-1|i-1>
#print(b_matrix)

b_matrix_Dg = b_matrix.T # creation operator, b_dagger
#print(b_matrix_Dg)

hop_matrix = np.zeros((EL_SITE_NUM, EL_SITE_NUM))
for i in range(EL_SITE_NUM):
    hop_matrix[(i+1)%EL_SITE_NUM, i] = 1 # <i+1|c*_(i+1) c_i|i> = 1
    hop_matrix[i, (i+1)%EL_SITE_NUM] = 1 # <i|c*_i c_(i+1)|i+1> = 1
hop_matrix *= (-1)*t
if EL_SITE_NUM == 2: hop_matrix *= 2 # Equiv. to larger size hopping matrix
#print(hop_matrix)

for i in range(EL_SITE_NUM):
    hop_matrix = np.kron(hop_matrix, np.eye(PH_MAX)) # Tensor Product
#print(hop_matrix)

bb_matrix = b_matrix_Dg @ b_matrix
#print(bb_matrix)

tot_matrix_size = EL_SITE_NUM * (PH_MAX ** EL_SITE_NUM)
phonon_matrix = np.zeros((tot_matrix_size, tot_matrix_size))
for i in range(EL_SITE_NUM):
    tmp_matrix = np.eye(EL_SITE_NUM)
    for j in range(EL_SITE_NUM):
        if j==i: tmp_matrix = np.kron(tmp_matrix, bb_matrix)
        else: tmp_matrix = np.kron(tmp_matrix, np.eye(PH_MAX))
    phonon_matrix += tmp_matrix
phonon_matrix *= omega
#print(phonon_matrix)

el_ph_matrix = np.zeros((tot_matrix_size, tot_matrix_size))
for i in range(EL_SITE_NUM):
    n_matrix = np.zeros((EL_SITE_NUM, EL_SITE_NUM)); n_matrix[i,i] = 1
    tmp_matrix = n_matrix
    for j in range(EL_SITE_NUM):
        if j==i: tmp_matrix = np.kron(tmp_matrix, (b_matrix_Dg + b_matrix))
        else: tmp_matrix = np.kron(tmp_matrix, np.eye(PH_MAX))
    el_ph_matrix += tmp_matrix
el_ph_matrix *= g
#print(el_ph_matrix)

H_matrix = hop_matrix + phonon_matrix + el_ph_matrix

eigvals, eigvecs = eigsh(H_matrix, k=3, which='SA')
print("Eigenvalues: "+str(eigvals))