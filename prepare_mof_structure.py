#!/usr/bin/env python

import numpy as np


def count_specie(sites):
    counts = {'O': 0, 'F': 0}
    for s in sites:
        counts[s.specie.name] += 1
    return counts


def hydroxylate_site(struct, rCrO, rCrOH=1.828, rOH=0.975, CrOH=110.4):
    struct = struct.copy()
    for i, s in enumerate(struct.sites):
        if s.specie.name == 'Cr':
            neighbors = struct.get_neighbors(s, rCrO, include_index=True)
            counts = count_specie(neighbors)
            assert(counts['O'] == 5)
            if len(neighbors) == 6:
                assert(counts['F'] == 1)
            elif len(neighbors) == 5:
                # add OH-
                vec = s.coords - neighbors[0][0].coords
                for n in neighbors[1:]:
                    vec += s.coords - n.coords
                vec = vec/np.linalg.norm(vec)
                struct.append('O', s.coords+vec*rCrOH, coords_are_cartesian=True)
                struct.append('H', s.coords+vec*(rCrOH+rOH), coords_are_cartesian=True)
            else:
                print('Cr [{}] has {} neighbors'.format(i, len(neighbors)))
                raise RuntimeError()
    return struct


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Hydroxylate open Cr sites.')
    parser.add_argument('--cif', action='store', required=True)
    parser.add_argument('-o', action='store', dest='output')
    parser.add_argument('--rCrO', action='store', default=2.2)
    args, unknown = parser.parse_known_args()
    return args


def write_gro(struct, filename, rCrO):
    with open(filename,'w') as fgro:
        fgro.write('Open Cr sites terminated by OH-')

        visited = np.zeros(len(struct))
        # Cr-O-H units
        for i, s in enumerate(struct.sites):
            if s.specie.name == 'Cr':
                neighbors = struct.get_neighbors(s, rCrO, include_index=True)
                counts = count_specie(neighbors)
                if counts['O'] == 6:
                    pass

        fconf.write("%5d\n" % (sum([len(i)*len(atomnameG[x]) for x,i in confG.items()])))
        count=0
        rid=count
        for atom in ('Si','Os','H','C','N'):
            rn=residuenameG[atom]
            an=atomnameG[atom]
            for molecule in confG[atom]:
                rid+=1
                for i,line in enumerate(molecule):
                    count+=1
                    fconf.write("%5d%5s%5s%5d%s" % (rid,rn,an[i],count,line))



if __name__ == '__main__':
    args = parse_args()
    print(args)

    from pymatgen.core import Structure
    struct = Structure.from_file(args.cif)

    struct_hydroxylated = hydroxylate_site(struct, rCrO=args.rCrO)

    if args.output is None:
        fname = args.cif.replace('.cif', '_hydroxylated.cif')
    else:
        fname = args.output
    # write_gro(struct_hydroxylated, filename=fname)
    struct_hydroxylated.to(filename=fname)
