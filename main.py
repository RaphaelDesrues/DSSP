"""Secondary Structure prediction using the DSSP method
"""


import os
import sys
import numpy as np


def check_file_exists(file_path):
    """Check if a file exists
    """
    if len(sys.argv) != 2:
        sys.exit("Il faut un fichier PDB")
    
    file_path = sys.argv[1]
    
    return file_path


def read_pdb(file_path):
    """Open a PDB file and output backbone atoms within a protein
    """
    
    atoms = ["C", "O", "N", "H"]
    position = []


    with open(file_path, 'r') as f_in:

        lines = f_in.readlines()

        for line in lines:

            if line.startswith("ATOM") and line[12:16].strip() in atoms:
                
                chain = float(line[21:22].strip())

                resid = float(line[22:26].strip())

                name = float(line[17:20].strip())

                x_coords = float(line[30:38].strip())
                y_coords = float(line[38:46].strip())
                z_coords = float(line[46:54].strip())

                position.append((x_coords, y_coords, z_coords))

        
            System.add_atom(chain, resid, name, position)


def dist(atom1: np.ndarray, atom2: np.ndarray) -> float:
    """Compute distance between 2 atoms"""

    d = np.sqrt(((atom1 - atom2)**2).sum())

    return d


def angle(atom1: np.ndarray, atom2: np.ndarray, atom3: np.ndarray) -> float:
    """Compute angle between 3 atoms"""
    
    v1 = atom1 - atom2
    v2 = atom3 - atom3

    angle = np.arccos((v1 * v2).sum()) / ((np.sqrt(v1**2).sum()) * (np.sqrt(v2**2).sum()))

    return angle


class Atom:
    """Docstring class"""

    def __init__(
        self,
        chain : str,
        resid: str,
        name: str,
        position: np.ndarray
    ):
        """Docstring

        Parameters
        ==========

        """

        self.chain = name
        self.resid = chain
        self.name = resid
        self.position = position


class System:
    """Docstring system"""

    def __init__(
        self,
        atoms: list,
        hbonds: list,
        nturn: list,
        helix: list,
        bridge: list
    ):
        """Docstring
        
        Parameters
        ==========
        
        """

        self.atoms = atoms
        self.hbonds = hbonds
        self.nturn = nturn
        self.helix = helix
        self.bridge = bridge


    def add_atom(
        self,
        name: str,
        chain: str,
        resid: str,
        position: np.ndarray
    ):
        """Docstring"""

        atom = Atom(name, chain, resid, position)
        self.atoms.append(atom)

        return atom
    

    def hbonds_calc(self, atom1, atom2, atom3):
        """Compute hbonds in whole protein"""

        
        # for atom1 in (self.atom.name in "N"):
        #     for self.atom.name in "O":
        #         dist = dist()
        #         angle = angle()

        #         if dist < 5.2 and angle < 63:
        #             self.hbonds.append([])



        # N_atoms = [atom1 for atom1 in self.atoms if atom1.name = "N"]
        # O_atoms = [atom2 for atom2 in self.atoms if atom2.name = "O"]

        # for atom1 in N_atoms:
        #     for atom2 in O_atoms:
        #         dist = dist(atom1, atom2)
        #         angle = angle(atom1, atom2, atom3)

        #         if dist < 5.2 and angle < 63:
        #             self.hbonds.append([atom1, atom2])

            









