"""Secondary Structure prediction using the DSSP method
"""


import os
import sys
import numpy as np
# import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # ou 'Agg', 'Qt5Agg', 'TkAgg' etc.
import matplotlib.pyplot as plt
import seaborn as sns
import time

import pymol
from pymol import cmd


# Constantes
Q1 = 0.42
Q2 = 0.20
DIM_F = 332
ENERGY_CUTOFF = -3


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
        hbonds: list[np.ndarray]
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


    def energy_hbond(self, atom1, atom2, atom3, atom4):
        """Compute energy in a H-bonding group
        
        atom1 = N
        atom2 = O
        atom3 = C
        atom4 = H
        """

        dist_ON = 1 / self.dist(atom1, atom2)
        dist_CH = 1 / self.dist(atom3, atom4)
        dist_OH = 1 / self.dist(atom2, atom4)
        dist_CN = 1 / self.dist(atom1, atom3)

        energy = Q1 * Q2 * DIM_F * (dist_ON + dist_CH - dist_OH - dist_CN)

        return energy


    def hbonds_calc(self):
        """Compute hbonds in whole protein"""

        chains = {atom.chain for atom in self.atoms}

        for chain in chains:

            N_atoms = [atom1 for atom1 in self.atoms if atom1.name == "N" and atom1.chain == chain]
            O_atoms = [atom2 for atom2 in self.atoms if atom2.name == "O" and atom2.chain == chain]
            H_atoms = [atom3 for atom3 in self.atoms if atom3.name == "H" and atom3.chain == chain]
            C_atoms = [atom4 for atom4 in self.atoms if atom4.name == "C" and atom4.chain == chain]


            # Create empty hbond boolean matrix
            self.hbonds.append(np.zeros((180, 180), dtype=bool))

            for atom1, atom3 in zip(N_atoms, H_atoms):

                for atom2, atom4 in zip(O_atoms, C_atoms):

                    # Compute O---N distance
                    dist = self.dist(atom1.position, atom2.position)

                    # Compute O---H-N angle
                    angle = self.angle(atom1.position, atom2.position, atom3.position)

                    if dist < 5.2 and angle < 63:

                        energy = self.energy_hbond(atom1.position, atom2.position, atom3.position, atom4.position) / 10

                        if energy < ENERGY_CUTOFF:
                            chain_index = chains.index(chain)
                            print(chain_index)
                            self.hbonds[chain_index][atom1.resid][atom2.resid] = True

                        else:
                            continue
                        
            # self.plot_hbonds()

            # Add nan values in diagonal
            # np.fill_diagonal(self.hbonds, np.nan)

  


    def plot_hbonds(self):
        """Plot a heatmap of the hbonds"""

        # Convertir la matrice de booléens en entiers (1 pour True, 0 pour False)
        hbond_matrix = self.hbonds.astype(int)

        resid_values = [atome.resid for atome in self.atoms]
        min_index = min(resid_values)
        max_index = max(resid_values)

        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        subset_matrix = hbond_matrix[min_index:max_index, min_index:max_index]

        sns.heatmap(hbond_matrix, cmap='coolwarm', annot=False, ax=axs[0])
        axs[0].set_title(f"Heatmap of hbonds")

        sns.heatmap(subset_matrix, cmap='coolwarm',  annot=False, ax=axs[1],
        xticklabels=list(range(min_index, max_index)), 
        yticklabels=list(range(min_index, max_index))
        )
        axs[1].set_title(f"Heatmap of hbonds (Subset)")

        # Sauvegarder et afficher
        plt.savefig(f"heatmap_combined_hbonds.png")
        plt.show()


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


    def nturn_calc(self, system):
        """Compute nturn from hbonds"""

        self.nturn = np.zeros((180, 180))

        for i in range(len(system.hbonds)):

            for j in range(len(system.hbonds)):
                
                if system.hbonds[i][j] == True:
                    
                    if (system.hbonds[i + 3][j] == True) or system.hbonds[i][j + 3] == True:
                        
                        self.nturn[i][j] = 3
                        

                    if (system.hbonds[i + 4][j] == True) or system.hbonds[i][j + 4] == True:

                        self.nturn[i][j] = 4
                        
                    
                    if (system.hbonds[i + 5][j] == True) or system.hbonds[i][j + 5] == True:

                        self.nturn[i][j] = 5
                        
                    else:
                        continue

        self.plot_heatmap(self.nturn, title = "nturn", display = False)


    def helices_calc(self):
        """Compute alpha-helices from nturns"""
        
        self.helix = np.zeros((180, 180))

        for i in range(len(self.nturn)):
            
            for j in range(len(self.nturn)):

                if self.nturn[i][j] == 4:
                    
                    self.helix[i, j] = 1 
        
        self.plot_heatmap(self.helix, title = "Helix", display = False)

        indices = []
        for i in range(len(self.helix)):
            for j in range(len(self.helix)):
                if self.helix[i, j] != 0:
                    indices.append([i, j])
        
        indices = list(set(element for sous_liste in indices for element in sous_liste))
        return indices


    def bridge_calc(self, system):
        """Compute bridges from Hbonds"""

        resid_list = []
        for i in range(len(system.hbonds)):
            first = i
            second = i + 1
            third = i + 2
            if third - first == 2:
                resid_list.append([first, second, third])


        self.bridge = np.zeros((180, 180))

        for i in range(len(resid_list) - 2):

            for j in range(len(resid_list) - 2):

                if any(elem in resid_list[j] for elem in resid_list[i]):
                    continue

                else:
                    a = resid_list[i]
                    b = resid_list[j]
                    cond_1 = (system.hbonds[a[0], b[1]]) and (system.hbonds[b[1], a[2]])
                    cond_2 = (system.hbonds[b[0], a[1]]) and (system.hbonds[a[1], b[2]])
                    cond_3 = (system.hbonds[a[1], b[1]]) and (system.hbonds[b[1], a[1]])
                    cond_4 = (system.hbonds[a[0], b[2]]) and (system.hbonds[b[0], a[2]])

                    if cond_1 or cond_2:
                        self.bridge[a[1], b[1]] = 1 # P


                    elif cond_3 or cond_4:
                        self.bridge[a[1], b[1]] = 2 # AP

        self.plot_heatmap(self.bridge, title = "Bridges", display = False)

        indices = []
        for i in range(len(self.bridge)):
            for j in range(len(self.bridge)):
                if self.bridge[i, j] != 0:
                    indices.append([i, j])
        
        indices = list(set(element for sous_liste in indices for element in sous_liste))
        # return indices

   
    def ladder_calc(self):
        """Compute beta-ladder from bridge"""

        self.ladder = np.zeros((180, 180))

        for i in range(1, len(self.bridge) - 1):

            for j in range(1, len(self.bridge) - 1):

                if self.bridge[i, j] != 0 and self.bridge[i, j] == self.bridge[i - 1, j + 1]:
                    self.ladder[i, j] = self.bridge[i, j]
                    self.ladder[i - 1, j + 1] = self.bridge[i - 1, j + 1]

                else:
                    continue

        self.plot_heatmap(self.ladder, title = "Ladder", display = False)

        

        
        indices = []
        for i in range(len(self.ladder)):
            for j in range(len(self.ladder)):
                if self.ladder[i, j] != 0:
                    indices.append([i, j])
        
        indices = list(set(element for sous_liste in indices for element in sous_liste))
        return indices


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


    def write_DSSP(self):
        """Write a DSSP file"""
        with open("DSSP_FINAL", "w") as f_out:
            # first line : PDB name, function, source 
            f_out.write(f"{sys.argv[1].split('.')[0]}\n")

            # Second line : Title of the acronyms for after
            f_out.write(f"{'B-sheet'} {'Bridge2'} {'Bridge1'} {'Chirality'} {'Bend'} {'5turn'} {'4turn'} {'3turn'}\n")

            # 3ème ligne : Sheet : A, B, C,...
            f_out.write(f"")

            # 4ème ligne : Bridge2 : A, B, C

            # 5ème ligne : Bridge1 : majuscule (anti//) et minuscule (//)

            # 6ème ligne : Chirality

            # 7ème ligne : Bend "S"

            # 8/9/10ème ligne : 5/4/3-turn


    def color_pymol(self, indices: list, color : str):
        '''Open PyMOL and color according to residue list'''

        # Ouvrir l'interface graphique de PyMOL
        pymol.finish_launching()

        cmd.load(f"{sys.argv[1]}")

        # Assurer que le chargement est terminé avant de colorer
        cmd.refresh()

        for index in indices:
            cmd.color(f"{color}", f"resi {index}")

        output_file = f"{sys.argv[1].split('.')[0]}_processed.pse"

        if os.path.exists(output_file):
            os.remove(output_file)

        cmd.save(output_file)
        print(f"SAVED {sys.argv[1].split('.')[0]}_processed.pse")

        # time.sleep(10)

        # cmd.quit()

    
    def plot_heatmap(self, matrix, title, display):
        '''Plot heatmaps'''

        sns.heatmap(matrix, cmap = 'coolwarm', annot = False)
        plt.title(f"Heatmap of {title}")

        plt.savefig(f"heatmap_{title}.png")

        if display == True:
            # plt.gca().invert_yaxis()
            plt.show()
            plt.close()

        plt.close()



def main():
    """Do main system"""
    # system = System()
    file_path = check_file_exists()
    system = read_pdb(file_path)
    system = remove_residues(system)

    # for atom in system.atoms:
    #     print(atom.chain, atom.resid, atom.name)


    system.hbonds_calc() 

    # dssp = Dssp(nturn = [],
    #             helix = [],
    #             bridge = [],
    #             ladder = [],
    #             sheet = None
    #             )

    # dssp.nturn_calc(system)
    # # dssp.helices_calc()
    # indices = dssp.helices_calc()
    # # dssp.color_pymol(indices, color = 'red')


    # dssp.bridge_calc(system)
    # # dssp.ladder_calc()
    # indices = dssp.ladder_calc()
    # # dssp.color_pymol(indices, color = 'blue')

    # dssp.write_DSSP()




if __name__ == "__main__":
    main()
