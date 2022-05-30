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
import matplotlib as mpl
import matplotlib.pyplot as plt

def bond_equal(bond1, bond2):
    sites = [bond1.site1, bond1.site2]
    if (bond2.site1 in sites) and (bond2.site2 in sites):
        return True
    else:
        return False

def bond_equiv(bond1, bond2):
    # 将bond两端的原子进行空间群对称操作后，会与原胞里的原子差若干个晶格常数（因为这是原胞），需要做些处理再判断等价与否，处理过程如下：
    ###### bond2可能与整数晶格有数值误差，例如某一分量为0.99999之类的，bond1没有这个情况
    ###########################################
    # 情况1：原胞的原子坐标0.00001，变换后为0.999999999
    # 情况2：原胞的原子坐标0.999999999，变换后还是0.9999999999
    # 情况3：原胞的原子坐标0.999999999，变换后变为1.0000000001
    ###########################################
    global tol
    # vec_a是第一种情况:bond1.site1对应bond2.site1
    # vec_b是第二种情况:bond1.site2对应bond2.site1
    # 给vec加around是处理数值的问题，比如: 1.0e+00实际是0.9999999999,取余数时会剩下1
    vec_a1 = np.around(bond1.site1.frac_coords - bond2.site1.frac_coords, 5)
    vec_a2 = np.around(bond1.site2.frac_coords - bond2.site2.frac_coords, 5)
    vec_b1 = np.around(bond1.site1.frac_coords - bond2.site2.frac_coords, 5)
    vec_b2 = np.around(bond1.site2.frac_coords - bond2.site1.frac_coords, 5)

    if (norm(vec_a1 - vec_a2) < tol) and (norm(np.abs(vec_a1) % np.array([1, 1, 1])) < tol) and (norm(np.abs(vec_a2) % np.array([1, 1, 1])) < tol):
        return True
    elif (norm(vec_b1 - vec_b2) < tol) and (norm(np.abs(vec_b1) % np.array([1, 1, 1])) < tol) and (norm(np.abs(vec_b2) % np.array([1, 1, 1])) < tol):
        return True
    else:
        return False
def site_order(site, struct):
    # 返回的原子序号与vesta显示的一致（从1开始）
    global tol
    n = 0
    for i in range(len(struct)):
        if site == struct[i]:
            return i+1
            break
        n += 1
    # 如果这个原子在原胞外，返回等价的原子序号与偏移晶格量
    if n == len(struct):
        for j in range(len(struct)):
            if norm(site.frac_coords % np.array([1, 1, 1]) - struct[j].frac_coords) < tol:
                return (j+1, (site.frac_coords // np.array([1, 1, 1])))
    raise Exception("A site has no correspondence.")

def get_all_bonds(struct):
################################ (2) get all bonds ####################################
    # neighbors of each atom, 1.8 is the largest carbon-carbon bond length
    nhbrs = struct.get_all_neighbors(r=1.8)
    bonds = []
    for i in range(len(struct)):
        for j in range(len(nhbrs[i])):
            bond_tmp = CovalentBond(struct[i], nhbrs[i][j])
            idx = 0
            for k in bonds:
                if bond_equal(k, bond_tmp):
                    idx += 1
            if idx == 0:
                bonds.append(bond_tmp)
    return bonds
################################ (2) get all bonds ####################################


def find_equivalent_bonds():
    global tol
    struct = Structure.from_file('POSCAR.vasp')
    struct_lattice = struct.lattice
############################### (1) get symmetry operations #############################
    lat=struct.lattice._matrix
    coords=struct.frac_coords
    spes = struct.atomic_numbers
    spg_spes = list(spes)
    spg_lat = [tuple(i) for i in lat]
    spg_coords = [tuple(i) for i in coords]
    sym = spglib.get_symmetry((spg_lat, spg_coords, spg_spes), symprec=tol)
    space_group = spglib.get_spacegroup((spg_lat, spg_coords, spg_spes), symprec=tol)
############################### (1) get symmetry operations #############################

    bonds = get_all_bonds(struct)

######################## (3) test: plot structure #######################
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # plt.xlabel('x')
    # plt.ylabel('y')
    # for i in range(len(bonds)):
    #     tmp1 = bonds[i].site1.coords
    #     tmp2 = bonds[i].site2.coords
    #     ax.plot([tmp1[0], tmp2[0]], [tmp1[1], tmp2[1]], [tmp1[2], tmp2[2]])
    # plt.show()
######################## (3) test: plot structure #######################

######################### (4) find equivalent bonds ###########################
    bonds_test = bonds.copy()
    #记录所有等价键的序号
    bond_sym_idx_all = []

    while True:
        # 跳出循环条件
        # print('bonds_test: {}'.format(bonds_test))
        if bonds_test == [False] * len(bonds):
            break
        # 记录某一类等价键序号
        bond_sym_idx = []
        # 得到第一个不是False的bond
        for t in range(len(bonds_test)):
            if bonds_test[t] != False:
                bond_origin_site1_frac_coords = bonds_test[t].site1.frac_coords.copy()
                bond_origin_site2_frac_coords = bonds_test[t].site2.frac_coords.copy()
                break
        # 根据不同对称性循环
        for i in range(len(sym['rotations'])):
            # 跳出循环条件
            if bonds_test == [False] * len(bonds):
                break

            site1_new_frac_coord = sym['rotations'][i].dot(bond_origin_site1_frac_coords) + sym['translations'][i]
            site1_new = PeriodicSite(6, lat.T.dot(site1_new_frac_coord), lattice=struct_lattice, coords_are_cartesian=True)
            site2_new_frac_coord = sym['rotations'][i].dot(bond_origin_site2_frac_coords) + sym['translations'][i]
            site2_new = PeriodicSite(6, lat.T.dot(site2_new_frac_coord), lattice=struct_lattice, coords_are_cartesian=True)
            bond_new = CovalentBond(site1_new, site2_new)
            # 每次筛选出和同类的bond，并将它们剔除出bonds_test(它们的位置由False取代)
            for j in range(len(bonds_test)):
                if bonds_test[j] == False:
                    continue
                if bond_equiv(bonds_test[j], bond_new):
                    bond_sym_idx.append(j)
                    bonds_test[j] = False

        bond_sym_idx_all.append(bond_sym_idx)

    # 输出所有等价的bond和它们对应的原子
    out = []
    out.append(space_group)
    for i in range(len(bond_sym_idx_all)):
        equivalent_bond_type = ['Type'+str(i+1)]
        for j in range(len(bond_sym_idx_all[i])):
            # print('bond number: {}; site1: {}; site2: {}'.format(bond_sym_idx_all[i][j],
            #                                                      site_order(bonds[bond_sym_idx_all[i][j]].site1, struct),
            #                                                      site_order(bonds[bond_sym_idx_all[i][j]].site2, struct)))
            equivalent_bond_type.append([site_order(bonds[bond_sym_idx_all[i][j]].site1, struct),\
                                        site_order(bonds[bond_sym_idx_all[i][j]].site2, struct)])
        out.append(equivalent_bond_type)
    return out
######################### (4) find equivalent bonds ###########################

def delete_edge_redundant_bonds(bond_edge_delete_all):
    # 删除多余bonds（边缘的bonds）
    global tol
    bond_edge_delete = []
    for i in range(len(bond_edge_delete_all)):
        for j in range(i+1, len(bond_edge_delete_all)):
            # 区分这两个bonds的两个端点哪个是元组（原胞外），哪个是整数（原胞内）
            # 这里规定site1是整数，site2是元组
            if isinstance(bond_edge_delete_all[i][0], int):
                bond1_site1 = bond_edge_delete_all[i][0]
                bond1_site2 = bond_edge_delete_all[i][1]
            if isinstance(bond_edge_delete_all[i][1], int):
                bond1_site2 = bond_edge_delete_all[i][0]
                bond1_site1 = bond_edge_delete_all[i][1]
            if isinstance(bond_edge_delete_all[j][0], int):
                bond2_site1 = bond_edge_delete_all[j][0]
                bond2_site2 = bond_edge_delete_all[j][1]
            if isinstance(bond_edge_delete_all[j][1], int):
                bond2_site2 = bond_edge_delete_all[j][0]
                bond2_site1 = bond_edge_delete_all[j][1]
            if (bond1_site1 == bond2_site2[0]) and (bond2_site1 == bond1_site2[0])\
            and norm(bond1_site2[1] + bond2_site2[1]) < tol:
                bond_edge_delete.append(bond_edge_delete_all[j])
    return bond_edge_delete

def eq_bonds():
    #所有bonds
    equiv_bonds = find_equivalent_bonds()

    # 所有需要被删除的晶格边缘多余的键，以用于TB模型
    for i in range(1, len(equiv_bonds)):
        bond_edge_delete_all = []
        for j in range(1, len(equiv_bonds[i])):
            if isinstance(equiv_bonds[i][j][0], tuple) or isinstance(equiv_bonds[i][j][1], tuple):
                bond_edge_delete_all.append(equiv_bonds[i][j])
        bond_edge_delete = delete_edge_redundant_bonds(bond_edge_delete_all)
        for j in bond_edge_delete:
            equiv_bonds[i].remove(j)
    return equiv_bonds

def main():
    # 保存equivalent bonds for TB model
    equiv_bonds = eq_bonds()
    with open('eqbonds.pickle', 'wb') as f:
        pickle.dump(equiv_bonds, f)

if __name__ == '__main__':
    global tol
    tol  = 1e-3
    main()







