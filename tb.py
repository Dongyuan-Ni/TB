import pickle
import numpy as np
from numpy.linalg import *
from pythtb import *
from pymatgen.io.cif import Structure
import spglib
from pymatgen.core.structure import IStructure
from pymatgen.core.bonds import CovalentBond
from pymatgen.core.sites import Site
from pymatgen.core.sites import PeriodicSite
import matplotlib.pyplot as plt

if __name__ == '__main__':
    with open('eqbonds.pickle', 'rb') as f:
        eq_bonds = pickle.load(f)
    print('# of equivalent bond types: {}'.format(len(eq_bonds) - 1))
    number_of_eq_bond_types = len(eq_bonds) - 1


################################ Testing Parameters ################################
    # Hopping parameters
    t1 = -1.8
    t2 = -2.1
    t3 = -1.6
    t4 = -2.2
    t5 = -3.2
################################ Testing Parameters ################################

    t = [t1, t2, t3, t4, t5]
    
    struct = Structure.from_file('POSCAR.vasp')
    lat = struct.lattice._matrix
    coords = struct.frac_coords
    spes = struct.atomic_numbers
    # tight-binding model
    model=tb_model(3, 3, lat, coords)
    # define hopping between orbitals
    for i in range(1, len(eq_bonds)):
        for j in range(1, len(eq_bonds[i])):
            if isinstance(eq_bonds[i][j][0], int) and isinstance(eq_bonds[i][j][1], int):
                model.set_hop(t[i-1], eq_bonds[i][j][0] - 1, eq_bonds[i][j][1] - 1, np.array([0, 0, 0]))
            if isinstance(eq_bonds[i][j][0], int) and isinstance(eq_bonds[i][j][1], tuple):
                model.set_hop(t[i-1], eq_bonds[i][j][0] - 1, eq_bonds[i][j][1][0] - 1, eq_bonds[i][j][1][1])
            if isinstance(eq_bonds[i][j][0], tuple) and isinstance(eq_bonds[i][j][1], int):
                model.set_hop(t[i-1], eq_bonds[i][j][1] - 1, eq_bonds[i][j][0][0] - 1, eq_bonds[i][j][0][1])


    # solve model on a path in k-space
    k=[
        [0.0000, 0.0000, 0.0000],
        [-0.2808, 0.2808, 0.2808],
        [-0.2144, 0.2144, 0.3472],
        [0.0000, 0.0000, 0.5000],
        [0.2500, 0.2500, 0.2500],
        [ 0.0000, 0.5000, 0.0000],
        [0.2808, 0.7192, -0.2808],
        [0.5000, 0.5000, -0.5000],
        [0.0000, 0.0000, 0.0000],
        [0.4336, -0.4336, 0.4336],
        [0.5000, 0.0000, 0.0000],
        [0.2500, 0.2500, 0.2500]
    ]

    (k_vec,k_dist,k_node)=model.k_path(k, 500)
    evals=model.solve_all(k_vec)

    # Dump bandstructure
    with open('tb.dat', 'w') as f:
        for i in range(evals.shape[0]):
            for j in range(len(evals[0, :])):
                f.write(str(k_dist[j])+' '+str(evals[i, j])+'\n')
            f.write('1  \n')
        ax.plot(k_dist, evals[i, :])
    # plot bandstructure
    # fig, ax = plt.subplots()
    # ax.plot(k_dist,evals[0,:])
    # for i in range(evals.shape[0]):
    #     ax.plot(k_dist, evals[i, :])
    # # ax.plot(k_dist,evals[1,:])
    # ax.set_xticks(k_node)
    # # ax.set_xticklabels(["$\Gamma$","K","M"])
    # ax.set_xlim(k_node[0],k_node[-1])
    # ax.set_ylim(-2, 2)
    # fig.savefig("band.png")
    
    # plot PBE bandstructure
    # d_ori = []
    # k_ori = []
    # e_ori = []
    # with open('eigen1.csv', 'r') as f:
    #     csv_reader = csv.reader(f)
    #     for row in list(csv_reader):
    #         d_ori.append(row)
    # d_ori = d_ori[:-3]
    # k_tmp = []
    # e_tmp = []
    # for i in range(len(d_ori)):
    #     if d_ori[i] != ['1']:
    #         k_tmp.append(float(d_ori[i][0].split()[0]))
    #         e_tmp.append(float(d_ori[i][0].split()[1]))
    #     else:
    #         k_ori.append(k_tmp)
    #         e_ori.append(e_tmp)
    #         k_tmp = []
    #         e_tmp = []
    #
    #
    # for i in range(len(e_ori)):
    #     for j in range(len(e_ori[i])):
    #         e_ori[i][j] -= 5.3072
    # for i in range(len(k_ori)):
    #     ax.plot(k_ori[i], e_ori[i], c='gray')

    # plot HSE bandstructure
    # d_ori = []
    # k_ori = []
    # e_ori = []
    # with open('hse1.csv', 'r') as f:
    #     csv_reader = csv.reader(f)
    #     for row in list(csv_reader):
    #         d_ori.append(row)
    # d_ori = d_ori[:-3]
    #
    # for i in range(len(d_ori)):
    #     if len(d_ori[i][0].split()) != 1:
    #         k_ori.append(float(d_ori[i][0].split()[0]))
    #         e_ori.append(float(d_ori[i][0].split()[1]))

    # for i in range(len(e_ori)):
    #     e_ori[i] -= 5.3072
    # ax.scatter(k_ori, e_ori, c='gray',s=1)
