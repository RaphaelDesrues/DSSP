"""Secondary Structure prediction using the DSSP method
"""


import os
import sys
import numpy as np
# import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # ou 'Agg', 'Qt5Agg', etc.
import matplotlib.pyplot as plt
import seaborn as sns


def check_file_exists():
        """Check if a file exists"""
        if len(sys.argv) != 2:
            sys.exit("Il faut un fichier PDB")
    
        file_path = sys.argv[1]
    
        return file_path


def read_pdb(file_path):
    """Open a PDB file and output backbone atoms within a protein"""

    atoms = ["C", "O", "N", "H"]
    
    system = System(atoms=None, hbonds=None)

    with open(file_path, 'r') as f_in:

        lines = f_in.readlines()

        for line in lines:

            if line.startswith("ATOM") and line[12:16].strip() in atoms:

                position = []

                chain = str(line[21:22].strip())

                resid = int(line[22:26].strip())

                name = str(line[12:16].strip())

                x_coords = float(line[30:38].strip())
                y_coords = float(line[38:46].strip())
                z_coords = float(line[46:54].strip())

                position.append((x_coords, y_coords, z_coords))

                system.add_atom(chain, resid, name, position)
    
    return system


def remove_residues(system):
    """Retire tous les résidus du système avec moins de 4 atomes."""

    residue_count = {}
    
    for atom in system.atoms:

        if atom.resid in residue_count:
            residue_count[atom.resid] += 1
        
        else:
            residue_count[atom.resid] = 1

    system.atoms = [atom for atom in system.atoms if residue_count[atom.resid] >= 4]

    return system
    

class Atom:
    """Docstring class"""

    def __init__(
        self,
        chain: str,
        resid: int,
        name: str,
        position: np.ndarray
    ):
        """Docstring

        Parameters
        ==========

        """

        self.chain = chain
        self.resid = resid
        self.name = name
        self.position = position


class System:
    """Docstring system"""

    def __init__(
        self,
        atoms: list,
        hbonds: np.ndarray,
    ):
        """Docstring
        
        Parameters
        ==========
        
        """

        if atoms is None:
            atoms = []
        if hbonds is None:
            hbonds = np.empty((0, 0))

        self.atoms = atoms
        self.hbonds = hbonds


    def add_atom(
        self,
        chain: str,
        resid: int,
        name: str,
        position: np.ndarray
    ):
        """Docstring""" 
        atom = Atom(chain, resid, name, position)
        self.atoms.append(atom)


    def dist(self, atom1: np.ndarray, atom2: np.ndarray) -> float:
        """Compute distance between 2 atoms"""
        atom1 = np.array(atom1)
        atom2 = np.array(atom2)

        d = np.sqrt(((atom1 - atom2)**2).sum())

        return d


    def angle(self, atom1: np.ndarray, atom2: np.ndarray, atom3: np.ndarray) -> float:
        """Compute angle between 3 atoms"""
        
        atom1 = np.array(atom1)
        atom2 = np.array(atom2)
        atom3 = np.array(atom3)

        

        v1 = (atom1 - atom2).flatten()
        v2 = (atom3 - atom2).flatten()

        # Compute scalar product
        scalar_product = np.dot(v1, v2)

        # Compute vector norms
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)

        angle = np.degrees(np.arccos(scalar_product / (norm_v1 * norm_v2)))

        return angle


    def hbonds_calc(self):
        """Compute hbonds in whole protein"""

        N_atoms = [atom1 for atom1 in self.atoms if atom1.name == "N"]
        O_atoms = [atom2 for atom2 in self.atoms if atom2.name == "O"]
        H_atoms = [atom3 for atom3 in self.atoms if atom3.name == "H"]

        # Create empty hbond boolean matrix
        self.hbonds = np.zeros((180, 180), dtype=bool)

        for atom1, atom3 in zip(N_atoms, H_atoms):

            for atom2 in (O_atoms):
                
                # Compute O---N distance
                dist = self.dist(atom1.position, atom2.position)

                # Compute O---H-N angle
                angle = self.angle(atom1.position, atom2.position, atom3.position)
                
                # print("DISTANCE", dist)
                # print('ANGLE', angle)
                
                if dist < 5.2 and angle < 63:
                    print("TRUE")
                    self.hbonds[atom1.resid][atom2.resid] = 1
                
                else:
                    # print("FALSE")
                    self.hbonds[atom1.resid][atom2.resid] = 0
        # print(self.hbonds)
        # Add nan values in diagonal
        # np.fill_diagonal(self.hbonds, np.nan)


    def plot_hbonds(self):
        """Plot a heatmap of the hbonds"""
        
        # print(self.hbonds[:50, :5])

        # sns.heatmap(self.hbonds, cmap='coolwarm', annot=True)
        # plt.title("Heatmap of hbonds")
        # plt.savefig("heatmap.png")
        # plt.close()
        # plt.show()

        # Convertir la matrice de booléens en entiers (1 pour True, 0 pour False)
        # hbond_matrix = self.hbonds.astype(int)
        # print(hbond_matrix)
        # sns.heatmap(hbond_matrix, cmap='coolwarm', annot=False)
        # plt.title("Heatmap of hbonds")
        # plt.savefig("heatmap.png")
        # plt.close()

        # Convertir la matrice de booléens en entiers (1 pour True, 0 pour False)
        hbond_matrix = self.hbonds.astype(int)

        # Afficher uniquement les lignes et colonnes de 25 à 160
        subset_matrix = hbond_matrix[25:161, 25:161]  # Slicing de 25 à 160 (inclusif)

        print(subset_matrix)  # Pour vérifier le contenu

        sns.heatmap(subset_matrix, cmap='coolwarm', annot=False)
        plt.title("Heatmap of hbonds")
        plt.savefig("heatmap.png")
        plt.close()


class Dssp:
    """Docstring system"""

    def __init__(
        self,
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

        self.nturn = nturn
        self.helix = helix
        self.bridge = bridge
        self.ladder = ladder
        self.sheet = sheet


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


def main():
    """Do main system"""

    file_path = check_file_exists()
    system = read_pdb(file_path)
    system = remove_residues(system)

    # for atom in system.atoms:
    #     print(atom.chain, atom.resid, atom.name)


    system.hbonds_calc()

    system.plot_hbonds()




if __name__ == "__main__":
    main()