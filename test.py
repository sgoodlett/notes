import optking
import psi4
import numpy as np


h2o = psi4.geometry(
	'''
	O
	H 1 0.946285801876863
	H 1 0.946285801876863 2 104.613063314222330
	'''
)

F = np.array(
	[[0.622805987002E+00, -.437211975749E-02, 0.302509582160E-01],
	 [-.437211975749E-02, 0.622805987002E+00, 0.302509582160E-01],
	 [0.302509582159E-01, 0.302509582159E-01, 0.181198353555E+00]])

hbar = 5.806484171
b_to_a = (0.5291772083)**-1
a_to_b = 0.5291772083
Eh_to_cm = 219474.6313708
F = F * Eh_to_cm
for i in range(2):
	F[:,i] *= b_to_a
	F[i,:] *= b_to_a

geom = np.asarray(h2o.geometry())

# Define intcoords using atom indices 0,1, or 2
intcoords = [optking.stre.Stre(0,1),optking.stre.Stre(0,2),optking.bend.Bend(2,0,1)]

# Create internal coordinates 
optking_coords = optking.intcosMisc.q_values(intcoords,geom)

# Compute B tensor given internal coordinates and cartesian coordinates
B_tensor = optking.intcosMisc.Bmat(intcoords,geom)
#B_tensor[2,:] = B_tensor[2,:] / b_to_a

# Create inverse mass arrays for H2O. Order matters.
massH = np.tile(np.array([1.007825032230]),3)
massO = np.tile(np.array([15.994914619570]),3)
mass = np.hstack((massO,massH,massH))
invmass_HHO = np.reciprocal(mass)

# G matrix
G = np.einsum('in,jn,n->ij', B_tensor, B_tensor, invmass_HHO)
G[:,2] *= b_to_a
G[2,:] *= b_to_a

# F = force constant matrix in internal coordinates (hessian)
GF = np.dot(G, F)

#Nitrogen Vals
GF_N = np.array(
    [[ 5.13723E+05, -1.20793E+04,  1.05408E+04],
     [-1.20884E+04,  5.13667E+05,  1.04044E+04],
     [-8.38628E+02, -1.13948E+03,  9.38646E+04]])

F_N = np.array(
    [[ 4.877666490402689E+05, -3.402909255277740E+03,  1.260024558228502E+04],
     [-3.402909255277740E+03,  4.877062385099230E+05,  1.247280561832061E+04],
     [ 1.260024558228502E+04,  1.247280561832061E+04,  3.992753726783524E+04]])

VR_1 = np.array(
    [[0.009762,  -7.664145,   7.647219],
     [0.013037,  -7.683767,  -7.627020],
     [4.698844,  -0.394243,   0.002902]])


VR = np.array(
        [[ -0.005501,  -0.065077,   0.065552],
         [ -0.005434,  -0.065250,  -0.065387],
         [  0.212845,   0.000316,   0.000045]])

freq = np.array([1779.4331856231,4112.2053598171,4210.3171790240])

# Diagonalize
labda, L = np.linalg.eig(GF)
print(labda,L)
#F_N_inv = np.linalg.inv(F_N)
#G_N = np.dot(GF_N,F_N_inv)

#Col sort
L[:,[2,0]] = L[:,[0,2]]
labda [[2,0]] = labda[[0,2]]
print(L,labda)
#print(5.806484171*(labda**0.5))
Lc = L.copy()

for i in range(3):
    temp = 0
    for j in range(3):
        for k in range(3):
            temp += L.T[i,k] * L.T[i,j] * F[k,j]
    temp = abs(temp)
    temp = np.sqrt(hbar*np.sqrt(abs(labda[i])) / temp)
    for j in range(3):
        Lc.T[i,j] *= temp
Lc_iT = np.linalg.inv(Lc).T

strang = ''
for i in range(3):
    for j in range(3):
        strang += ' {} '.format(GF_N[i,j] - GF[i,j])
    strang += '\n'

#print(f'F = \n{F}\n\nG = \n{G}\n\nF_N = \n{F_N}\n\nG_N = \n{G_N}\n\nGF = \n{GF}\n\nGF_N = \n{GF_N}\n\nGF / GF_N = \n{strang}\n\n')

print(f'Lc_iT = \n{Lc_iT}\n\nVR_iT = \n{VR_1}\n\nm = \n{Lc_iT/VR_1}')

