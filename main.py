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
        bridge: dict,
        ladder: dict,
        sheet: list
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
        self.ladder = ladder
        self.sheet = sheet


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
    

    def hbonds_calc(self):
        """Compute hbonds in whole protein"""

        
        # for atom1 in (self.atom.name in "N"):
        #     for self.atom.name in "O":
        #         dist = dist()
        #         angle = angle()

        #         if dist < 5.2 and angle < 63:
        #             self.hbonds.append([])



        N_atoms = [atom1 for atom1 in self.atoms if atom1.name == "N"]
        O_atoms = [atom2 for atom2 in self.atoms if atom2.name == "O"]
        H_atoms = [atom3 for atom3 in self.atoms if atom3.name == "H"]

        # for atom1 in N_atoms:

        #     for atom2 in O_atoms:

        #         dist = dist(atom1.position, atom2.position)

        #         atom3 = next((atom3 for atom3 in H_atoms if atom3.resid == atom1.resid), None)
        #         angle = angle(atom1.position, atom2.position, atom3.position)

        #         if dist < 5.2 and angle < 63:
        #             self.hbonds.append([atom1.resid, atom2.resid])


        for atom1, atom2, atom3 in zip(N_atoms, O_atoms, H_atoms):

            dist = dist(atom1.position, atom2.position)
            angle = angle(atom1.position, atom2.position, atom3.position)

            if dist < 5.2 and angle < 63:
                self.hbonds.append([atom1.resid, atom2.resid])


    def nturn_calc(self):
        """Compute nturn from hbonds"""

        num_test = []

        for i in range(len(self.hbonds)):

            num_test.append(self.hbonds[i] + 3)
            num_test.append(self.hbonds[i] + 4)
            num_test.append(self.hbonds[i] + 5)

            if self.hbonds[i][1] in num_test:

                self.nturn.append(self.hbonds[i][0])


    def helices_calc(self):
        """Compute alpha-helices from nturns"""

        for turn, turn_plus in zip(self.nturn, self.nturn[1:]):

            if turn + 1 == turn_plus:

                self.helices.append(turn)


    def bridge_calc(self):
        """Compute bridges from Hbonds"""

        premier_elem = [a[0] for a in self.hbonds]
        deuxieme_elem = [b[1] for b in self.hbonds]

        for i_moins, i, i_plus in zip(premier_elem,
                                      premier_elem[1:],
                                      premier_elem[2:]):
            
            for j_moins, j, j_plus in zip(deuxieme_elem,
                                          deuxieme_elem[1:],
                                          deuxieme_elem[2:]):
                
                cond_1 = [i_moins, j] in self.hbonds 
                cond_2 = [j, i_plus] in self.hbonds
                cond_3 = [j_moins, i] in self.hbonds
                cond_4 = [i, j_plus] in self.hbonds
                cond_5 = [i, j] in self.hbonds
                cond_6 = [j, i] in self.hbonds
                cond_7 = [i_moins, j_plus] in self.hbonds
                cond_8 = [j_moins, i_plus] in self.hbonds

                if (cond_1 and cond_2) or (cond_3 and cond_4):
                    self.bridge[i] = "P"

                
                elif (cond_5 and cond_6) or (cond_7 and cond_8):
                    self.bridge[i] = "AP"
                
                else:
                    self.bridge[i] = ""

    
    def ladder_calc(self):
        """Compute beta-ladder from bridge"""

        for key, value in self.bridge.items():

            if key == 1:
                temp = value
                continue

            else:
                if value == temp:

                    if key == 2:
                        self.ladder.append(1)

                    self.ladder.append(key)
                    temp = value
                
                else:
                    temp = value
                    continue


    def sheet_calc(self):
        """Compute beta-sheet from ladder"""

        for i, j in zip(self.ladder, self.ladder[1:]):

            if i + 1 == j and len(self.sheet) < 2:

                self.sheet.append(i)
                self.sheet.append(j)

            else:

                self.sheet.append(j)


    def bend_calc(self):
        """Compute bend from alpha-Carbons"""

        c_alpha = [c_atom for c_atom in self.atoms if c_atom.name == "CA"]

        