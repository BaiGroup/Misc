#!/usr/bin/env python

from numba import jit
import numpy as np
import sys


@jit(nopython=True)
def calculate_all_pair_distances(positions, cell):
    N = positions.shape[0]
    rsq = np.zeros((N, N), dtype=np.float_)
    vector = np.zeros((N, N, 3), dtype=np.int32)
    for i in range(N):
        for j in range(N):
            if j <= i:
                continue
            r = positions[j] - positions[i]
            vector[i, j] = np.rint(r / cell)
            vector[j, i] = -vector[i, j]
            r -= vector[i, j] * cell
            rsq[i, j] = np.dot(r, r)
            rsq[j, i] = rsq[i,j]
    return rsq, vector


def clean_up_valency(molDef, moltyp, neighbor):
    expected_valency = {
        molDef['H']: 1,
        molDef['C']: 4,
        molDef['N']: 3,
        molDef['O']: 2,
        molDef['Si']: 4,
    }

    corrected = []
    to_do = []

    for i, nb in enumerate(neighbor):
        if len(nb) == expected_valency[moltyp[i]]:
            corrected.append(i)
            to_do.append(i)

    start = 0
    if len(to_do) > start:
        end = len(to_do)
        for at in to_do[start:end]:
            for nb in neighbor[at]:
                if nb not in corrected and \
                   len(neighbor[nb])>expected_valency[moltyp[nb]]:
                    extra_neighbor = []
                    for nbsnb in neighbor[nb]:
                        if nbsnb not in corrected:
                            extra_neighbor.append(nbsnb)
                    if len(neighbor[nb])-len(extra_neighbor) == \
                       expected_valency[moltyp[nb]]:
                        for en in extra_neighbor:
                            remove_neighbor(neighbor[en], nb)
                            remove_neighbor(neighbor[nb], en)
                        to_do.append(nb)
        start = end


def find_silane(molDef, moltyp, coord, neighbor, rsq):
    '''Finds NH2-CH2-CH2-CH2-Si-O3.

    H atoms in CH2 are removed (united atom), O atoms bonded to Si are
duplicated (since they may be shared with a neighboring mono-silane moiety).

    '''

    print('Finding silanes')
    HDef = molDef['H']
    ODef = molDef['O']
    SiDef = molDef['Si']
    CDef = molDef['C']
    NDef = molDef['N']
    nSilane = 0
    coordSilane = []
    for iN in range(len(moltyp)):
        if moltyp[iN] == NDef:
            moltyp[iN] = 0
            print('N: {}'.format(iN), end='')
            coordSilane.append(coord[iN])

            H_idx = find_neighbors(iN, neighbor, moltyp, HDef, 2, rsq)
            for iH in H_idx:
                moltyp[iH] = 0
                coordSilane.append(coord[iH])
                print('; H: {}'.format(iH), end='')

            iCur = iN
            for jC in range(3):  # find 3 CH2 units
                iC = (find_neighbors(iCur, neighbor, moltyp, CDef, 1, rsq))[0]
                moltyp[iC] = 0
                coordSilane.append(coord[iC])
                print('; C: {}'.format(iC), end='')
                H_idx = find_neighbors(iC, neighbor, moltyp, HDef, 2, rsq)
                for iH in H_idx:
                    moltyp[iH] = 0
                    print('; H: {}'.format(iH), end='')
                iCur = iC

            iSi = (find_neighbors(iCur, neighbor, moltyp, SiDef, 1, rsq))[0]
            moltyp[iSi] = 0
            coordSilane.append(coord[iSi])
            print('; Si: {}'.format(iSi), end='')

            O_idx = find_neighbors(iSi, neighbor, moltyp, ODef, 3, rsq)
            for iO in O_idx:
                # do not set moltyp[iO] = 0
                coordSilane.append(coord[iO])
                print('; O: {}'.format(iO), end='')

            nSilane += 1
            print()
    if nSilane > 1:
        molDef['silane'] = len(molDef)+1
    return (np.array(coordSilane), nSilane)


def find_neighbors(i, neighbor, moltyp, target_moltyp, check=None, rsq=None):
    '''Find unvisited neighbors of atom i.

    If check is given and a positive number, then return the "check" number of
    nearest neighbors. If check is a negative number, then trigger an error if
    the number of neighbors does not match.

    '''

    targets = []
    for j in neighbor[i]:
        if moltyp[j] == target_moltyp:
            targets.append(j)
    targets = np.int32(targets)

    if check is not None:
        if check <= 0 or len(targets) <= check:
            assert len(targets) == abs(check), \
                'Atom {} bonded to {} neighbors of moltyp {}. Targets: {}; Neighbors: {}'.format(i+1, len(targets), target_moltyp, str(targets), str(neighbor[i]))
        else:
            print('Fixing: atom {} bonded to {} neighbors of moltyp {}. Targets: {}; Neighbors: {}'.format(i+1, len(targets), target_moltyp, str(targets), str(neighbor[i])))
            rsq_target = rsq[i, targets]
            targets_idx = np.argpartition(rsq_target, check)[:check]
            targets = targets[targets_idx]

    return targets


def find_silanol(molDef, moltyp, coord, neighbor, rsq):
    '''Find H-O-Si-O3.

    O atoms bonded to Si are duplicated (since they may be shared with a
neighboring silanol moiety)

    '''

    print('Finding silanols')
    HDef = molDef['H']
    ODef = molDef['O']
    SiDef = molDef['Si']
    nSilanol = 0
    coordSilanol = []
    for iH in range(len(moltyp)):
        if moltyp[iH] == HDef:
            moltyp[iH] = 0
            coordSilanol.append(coord[iH])
            print('H: {}'.format(iH), end='')

            iO = (find_neighbors(iH, neighbor, moltyp, ODef, 1, rsq))[0]
            moltyp[iO] = 0
            coordSilanol.append(coord[iO])
            print('; O: {}'.format(iO), end='')

            iSi = (find_neighbors(iO, neighbor, moltyp, SiDef, 1, rsq))[0]
            moltyp[iSi] = 0
            coordSilanol.append(coord[iSi])
            print('; Si: {}'.format(iSi), end='')

            O_idx = find_neighbors(iSi, neighbor, moltyp, ODef, 3, rsq)
            for iO in O_idx:
                # do not set moltyp[iO] = 0
                coordSilanol.append(coord[iO])
                print('; O: {}'.format(iO), end='')

            nSilanol += 1
            print()
    if nSilanol > 1:
        molDef['silanol'] = len(molDef)+1
    return (np.array(coordSilanol), nSilanol)


def read_pdb(file_pdb):
    box = ''
    nType = {}  # number of atoms of each element
    molDef = {}  # element -> molecule type mapping
    moltyp = []
    coord = []
    neighbor = None
    maxr = [-1000, -1000, -1000]
    minr = [1000, 1000, 1000]

    for lineStr in open(file_pdb):
        lineStr = lineStr.strip()
        if len(lineStr) == 0:
            continue

        line = lineStr.split()
        field = lineStr[:6].strip()

        if field == "CRYST1":
            box = np.float_([float(line[i+1]) for i in range(3)])
        elif field == "ATOM":
            try:
                nType[line[-1]] += 1
            except KeyError:
                nType[line[-1]] = 1
                molDef[line[-1]] = len(molDef)+1
            moltyp.append(molDef[line[-1]])

            coord.append([float(line[5+i]) for i in range(3)])  # VERIFY!
            for i in range(3):
                if coord[-1][i] > maxr[i]:
                    maxr[i] = coord[-1][i]
                if coord[-1][i] < minr[i]:
                    minr[i] = coord[-1][i]
        elif field == "CONECT":
            if neighbor is None:
                neighbor = [[] for i in moltyp]
            iAtom = int(lineStr[6:11])  # VERIFY: Aten is non-standard
            rest = [lineStr[i:i+5] for i in range(11, len(lineStr), 5)]
            for i in rest:
                jAtom = int(i)  # VERIFY: Aten is non-standard
                if jAtom not in neighbor[iAtom]:
                    neighbor[iAtom].append(jAtom)
                    neighbor[jAtom].append(iAtom)
    coord = np.float_(coord)
    return (box, nType, molDef, moltyp, coord, neighbor, maxr, minr)


def remove_neighbor(neighbor, item):
    for i, n in enumerate(neighbor):
        if n == item:
            del neighbor[i]
            return


def write_fort77(file_fort77, box, molDef, moltyp, coord, coordSilane, nSilane,
                 coordSilanol, nSilanol):
    fou = open(file_fort77, "w")
    fou.write('{} {} {}\n'.format(*box))

    for i in moltyp:
        if i != 0:
            fou.write(str(i)+' ')
    if nSilanol > 0:
        fou.write((str(molDef['silanol'])+' ')*nSilanol)
    if nSilane > 0:
        fou.write((str(molDef['silane'])+' ')*nSilane+'\n')

    # each Si-O-H removes 3 counts from the number of molecules and add 1
    # count to it, etc.
    fou.write('1 '*(len(moltyp)-2*nSilanol-12*nSilane)+'\n')

    dispz = np.float_([0.0,0.0,0.0])  # newz-(maxr[2]+minr[2])/2
    for i, mt in enumerate(moltyp):
        if mt != 0:
            print(coord[i])
            fou.write("{:9.5f} {:9.5f} {:9.5f} {:7.3f}\n".format(
                *(coord[i]+dispz), 0.0))

    for i in range(6*nSilanol):
        fou.write("{:9.5f} {:9.5f} {:9.5f} {:7.3f}\n".format(
            *(coordSilanol[i]+dispz), 0.0))

    for i in range(10*nSilane):
        fou.write("{:9.5f} {:9.5f} {:9.5f} {:7.3f}\n".format(
            *(coordSilane[i]+dispz), 0.0))

    fou.close()


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: {} /path/to/pdb /path/to/mcccs\n".format(sys.argv[0]))
        quit()

    newz = 45.0

    (box, nType, molDef, moltyp, coord, neighbor,
     maxr, minr) = read_pdb(sys.argv[1])
    print(nType)

    rsq, vector = calculate_all_pair_distances(coord, box)

    clean_up_valency(molDef, moltyp, neighbor)

    (coordSilane, nSilane) = find_silane(molDef, moltyp, coord, neighbor, rsq)
    (coordSilanol, nSilanol) = find_silanol(molDef, moltyp, coord, neighbor,
                                            rsq)
    write_fort77(sys.argv[2], box, molDef, moltyp, coord, coordSilane, nSilane,
                 coordSilanol, nSilanol)

    nType['silanol'] = nSilanol
    nType['silane'] = nSilane
    nType['Si'] -= nSilanol + nSilane
    nType['O'] -= nSilanol
    nType['H'] -= nSilanol + 8*nSilane
    nType['C'] -= 3*nSilane
    nType['N'] -= nSilane
    print(nType)
    print(molDef)
    print('max: {}; min: {}'.format(maxr,minr))
