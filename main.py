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
import pandas as pd
import time

import pymol
from pymol import cmd


# Constantes
Q1 = 0.42
Q2 = 0.20
DIM_F = 332
ENERGY_CUTOFF = -0.5


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

                resname = str(line[17:20].strip())

                x_coords = float(line[30:38].strip())
                y_coords = float(line[38:46].strip())
                z_coords = float(line[46:54].strip())

                position.append((x_coords, y_coords, z_coords))

                system.add_atom(chain, resid, name, position, resname)
    
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
        position: np.ndarray,
        resname: str
    ):
        """Docstring

        Parameters
        ==========

        """

        self.chain = chain
        self.resid = resid
        self.name = name
        self.position = position
        self.resname = resname


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
        position: np.ndarray,
        resname: str
    ):
        """Docstring""" 
        atom = Atom(chain, resid, name, position, resname)
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
        chains = sorted(chains)

        # Create empty hbond boolean matrix
        self.hbonds = {}

        for chain in chains:

            N_atoms = [atom1 for atom1 in self.atoms if atom1.name == "N" and atom1.chain == chain]
            O_atoms = [atom2 for atom2 in self.atoms if atom2.name == "O" and atom2.chain == chain]
            H_atoms = [atom3 for atom3 in self.atoms if atom3.name == "H" and atom3.chain == chain]
            C_atoms = [atom4 for atom4 in self.atoms if atom4.name == "C" and atom4.chain == chain]
            
            chain_index = (np.zeros((180, 180), dtype=bool))

            for atom1, atom3 in zip(N_atoms, H_atoms):

                for atom2, atom4 in zip(O_atoms, C_atoms):

                    # Compute O---N distance
                    dist = self.dist(atom1.position, atom2.position)

                    # Compute O---H-N angle
                    angle = self.angle(atom1.position, atom2.position, atom3.position)

                    if dist < 5.2 and angle < 63:

                        energy = self.energy_hbond(atom1.position, atom2.position, atom3.position, atom4.position) / 10

                        if energy < ENERGY_CUTOFF:

                            chain_index[atom1.resid][atom2.resid] = True

                        else:
                            continue
            
            self.hbonds[chain] = chain_index
            # print(f"Nombre de True Chaine {chain} : {np.sum(chain_index)}")
        
        # print(len(self.hbonds))
        # self.plot_hbonds()

            

    def plot_hbonds(self):
        """Plot a heatmap of the hbonds"""

        chains = {atom.chain for atom in self.atoms}
        chains = sorted(chains)

        if len(chains) > 1:
            
            fig, axs = plt.subplots(1, len(chains), figsize=(12, 6))
            axs = axs.flatten()  # Aplatir les axes pour itérer facilement dessus

            for i, chain in enumerate(chains):

                resid_values = [atome.resid for atome in self.atoms if atome.chain == chain]
                min_index = min(resid_values)
                max_index = max(resid_values)

                print(f"Chaine {chain}, min_index: {min_index}, max_index: {max_index}")
                subset_matrix = self.hbonds[chain][min_index:max_index + 1, min_index:max_index + 1]

                sns.heatmap(subset_matrix, cmap='coolwarm',  annot=False, ax=axs[i],
                            xticklabels=list(range(min_index, max_index + 1)), 
                            yticklabels=list(range(min_index, max_index + 1))
                )
                axs[i].set_title(f"Heatmap of hbonds Chain {chain}")

            plt.tight_layout()
            plt.show()

        else:
            # num_true = np.sum(self.hbonds[0])
            # print(f"Nombre de True dans la matrice des liaisons hydrogène : {num_true}")
            resid_values = [atome.resid for atome in self.atoms]
            min_index = min(resid_values)
            max_index = max(resid_values)

            subset_matrix = self.hbonds[chains[0]][min_index:max_index, min_index:max_index]

            sns.heatmap(subset_matrix, cmap='coolwarm',  annot=False,
                        xticklabels=list(range(min_index, max_index)), 
                        yticklabels=list(range(min_index, max_index))
            )
            
            plt.title = (f"Heatmap of hbonds Chain A")

            # Sauvegarder et afficher
            # plt.savefig(f"heatmap_combined_hbonds.png")
            plt.show()


class Dssp:
    """Docstring system"""

    def __init__(
        self,
        system,
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

        self.system = system
        self.nturn = nturn
        self.helix = helix
        self.bridge = bridge
        self.ladder = ladder
        self.sheet = sheet


    def nturn_calc(self):
        """Compute nturn from hbonds"""

        chains = {atom.chain for atom in self.system.atoms}
        chains = sorted(chains)

        self.nturn = {}

        for chain in chains:

            hbonds_temp = self.system.hbonds[chain]

            self.nturn[chain] = np.zeros((180, 180))

            for i in range(len(hbonds_temp)):

                for j in range(len(hbonds_temp)):

                    if hbonds_temp[i][j] == True:

                        if (hbonds_temp[i + 3][j] == True) or hbonds_temp[i][j + 3] == True:

                            self.nturn[chain][i][j] = 3


                        if (hbonds_temp[i + 4][j] == True) or hbonds_temp[i][j + 4] == True:

                            self.nturn[chain][i][j] = 4


                        if (hbonds_temp[i + 5][j] == True) or hbonds_temp[i][j + 5] == True:

                            self.nturn[chain][i][j] = 5

                        else:
                            continue
                    
        # self.plot_heatmap(matrix = self.nturn, title = "nturn", display = False)


    def helices_calc(self):
        """Compute alpha-helices from nturns"""
        
        chains = {atom.chain for atom in self.system.atoms}
        chains = sorted(chains)

        self.helix = {}
        
        for chain in chains:
            
            self.helix[chain] = np.zeros((180, 180))

            for i in range(len(self.nturn[chain])):

                for j in range(len(self.nturn[chain])):

                    if self.nturn[chain][i][j] == 4:

                        self.helix[chain][i, j] = 1 

        # self.plot_heatmap(self.helix, title = "Helix", display = False)


    def bridge_calc(self):
        """Compute bridges from Hbonds"""

        chains = {atom.chain for atom in self.system.atoms}
        chains = sorted(chains)

        self.bridge = {}

        for chain in chains:

            hbonds_temp = self.system.hbonds[chain]

            resid_list = []
            for i in range(len(hbonds_temp)):
                first = i
                second = i + 1
                third = i + 2
                if third - first == 2:
                    resid_list.append([first, second, third])


            self.bridge[chain] = np.zeros((180, 180))

            for i in range(len(resid_list) - 2):

                for j in range(len(resid_list) - 2):

                    if any(elem in resid_list[j] for elem in resid_list[i]):
                        continue

                    else:
                        a = resid_list[i]
                        b = resid_list[j]
                        cond_1 = (hbonds_temp[a[0], b[1]]) and (hbonds_temp[b[1], a[2]])
                        cond_2 = (hbonds_temp[b[0], a[1]]) and (hbonds_temp[a[1], b[2]])
                        cond_3 = (hbonds_temp[a[1], b[1]]) and (hbonds_temp[b[1], a[1]])
                        cond_4 = (hbonds_temp[a[0], b[2]]) and (hbonds_temp[b[0], a[2]])

                        if cond_1 or cond_2:
                            self.bridge[chain][a[1], b[1]] = 1 # P


                        elif cond_3 or cond_4:
                            self.bridge[chain][a[1], b[1]] = 2 # AP

        self.plot_heatmap(self.bridge, title = "Bridges", display = False)

   
    def ladder_calc(self):
        """Compute beta-ladder from bridge"""

        chains = {atom.chain for atom in self.system.atoms}
        chains = sorted(chains)

        self.ladder = {}

        for chain in chains:

            self.ladder[chain] = np.zeros((180, 180))

            for i in range(1, len(self.bridge[chain]) - 1):

                for j in range(1, len(self.bridge[chain]) - 1):

                    if self.bridge[chain][i, j] != 0 and self.bridge[chain][i, j] == self.bridge[chain][i - 1, j + 1]:
                        self.ladder[chain][i, j] = self.bridge[chain][i, j]
                        self.ladder[chain][i - 1, j + 1] = self.bridge[chain][i - 1, j + 1]

                    else:
                        continue
        
        self.plot_heatmap(self.ladder, title = "Ladder", display = False)


    def sheet_calc(self):
        """Compute beta-sheet from ladder"""

        chains = {atom.chain for atom in self.system.atoms}
        chains = sorted(chains)

        self.ladder = {}

        for chain in chains:

            for i, j in zip(self.ladder[chain], self.ladder[chain][1:]):

                if i + 1 == j and len(self.sheet[chain]) < 2:

                    self.sheet[chain].append(i)
                    self.sheet[chain].append(j)

                else:

                    self.sheet[chain].append(j)

        print(len(self.sheet))


    def bend_calc(self):
        """Compute bend from alpha-Carbons"""

        c_alpha = [c_atom for c_atom in self.atoms if c_atom.name == "CA"]


    # def write_DSSP(self):
    #     """Write a DSSP file"""
    #     with open("DSSP_FINAL", "w") as f_out:
    #         # first line : PDB name, function, source 
    #         f_out.write(f"{sys.argv[1].split('.')[0]}\n")

    #         # Second line : Title of the acronyms for after
    #         f_out.write(f"{'B-sheet'} {'Bridge2'} {'Bridge1'} {'Chirality'} {'Bend'} {'5turn'} {'4turn'} {'3turn'}\n")

    #         chains = {atom.chain for atom in self.system.atoms}
    #         chains = sorted(chains)

    #         for chain in chains:

    #             for i in range(len(self.system.atoms)):
                
    #                 # 3ème ligne : Sheet : A, B, C,...
                    
    #                 f_out.write(f"{self.bridge[i]}")

    #                 # 4ème ligne : Bridge2 : A, B, C

    #                 # 5ème ligne : Bridge1 : majuscule (anti//) et minuscule (//)

    #                 # 6ème ligne : Chirality

    #                 # 7ème ligne : Bend "S"

    #                 # 8/9/10ème ligne : 5/4/3-turn

    def write_DSSP(self):
        """Write a DSSP file"""

        df_dssp = pd.DataFrame(columns=['Chain', 'Sequence', 'Name', 'Sheet', 'Bridge2', 'Bridge1', 'Helix', '5-turn', '4-turn', '3-turn'])

        chains = {atom.chain for atom in self.system.atoms}
        chains = sorted(chains)

        for chain in chains:

            df = pd.DataFrame(columns=['Chain', 'Sequence', 'Name', 'Sheet', 'Bridge2', 'Bridge1', 'Helix', '5-turn', '4-turn', '3-turn'])
            
            resid_values = [atome.resid for atome in self.system.atoms if atome.chain == chain]        
            
            df['Sequence'] = np.arange(1, max(resid_values) + 1)
            df['Chain'] = chain

            previous_resid = 0
            i = 1
            for atome in self.system.atoms:
                if atome.chain == chain and atome.resid != previous_resid:
                    # print(atome.resid, atome.resname)
                    # print(df['Sequence'] == atome.resid)
                    df.loc[i, 'Name'] = atome.resname
                    previous_resid = atome.resid
                    i += 1


            df['Sheet'] = 0
            df['Bridge2'] = 0
            

            for i in range(max(resid_values)):
                # df.loc[i, 'Bridge2'] = self.ladder[chain][self.ladder[chain] != 0][0] if np.any(self.ladder[chain] != 0) else 0
                df.loc[i, 'Bridge1'] = next((x for x in self.bridge[chain][i] if x != 0), 0)
                df.loc[i, 'Helix'] = next((x for x in self.helix[chain][i] if x != 0), 0)
                df.loc[i, '5-turn'] = next((x for x in self.nturn[chain][i] if x == 5), 0)
                df.loc[i, '4-turn'] = next((x for x in self.nturn[chain][i] if x == 4), 0)
                df.loc[i, '3-turn'] = next((x for x in self.nturn[chain][i] if x == 3), 0)


            df_dssp = pd.concat([df_dssp, df])
        
        
        
        df_dssp.to_csv('df_dssp.txt', sep='\t', index=False, header=False)
        # Ajouter manuellement l'en-tête (avec tabulation mais sans tabulation dans les titres)
        with open('df_dssp.txt', 'r') as f:
            content = f.read()

        # L'en-tête avec des tabulations
        header = 'Chain Sequence Name Sheet Bridge2 Bridge1 Helix 5-turn 4-turn 3-turn\n'

        # Réécrire le fichier avec l'en-tête et les données
        with open('df_dssp.txt', 'w') as f:
            f.write(header + content)
        
        print(df_dssp)


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

        chains = {atom.chain for atom in self.system.atoms}
        chains = sorted(chains)
        # print(len(chains))

        if len(chains) > 1:

            fig, axs = plt.subplots(1, len(chains), figsize=(12, 6))
            axs = axs.flatten()  # Aplatir les axes pour itérer facilement dessus

            for i, chain in enumerate(chains):

                resid_values = [atome.resid for atome in self.system.atoms if atome.chain == chain]
                min_index = min(resid_values)
                max_index = max(resid_values)

                # print(f"Chaine {chain}, min_index: {min_index}, max_index: {max_index}")
                subset_matrix = matrix[chain][min_index:max_index + 1, min_index:max_index + 1]

                sns.heatmap(subset_matrix, cmap='coolwarm',  annot=False, ax=axs[i],
                            xticklabels=list(range(min_index, max_index + 1)), 
                            yticklabels=list(range(min_index, max_index + 1))
                )
                axs[i].set_title(f"Heatmap of {title} Chain {chain}")

            if display == True:
                plt.tight_layout()
                plt.show()

        else:
            # num_true = np.sum(self.hbonds[0])
            # print(f"Nombre de True dans la matrice des liaisons hydrogène : {num_true}")
            resid_values = [atome.resid for atome in self.system.atoms]
            min_index = min(resid_values)
            max_index = max(resid_values)

            subset_matrix = matrix[chains[0]][min_index:max_index, min_index:max_index]

            sns.heatmap(subset_matrix, cmap='coolwarm',  annot=False,
                        xticklabels=list(range(min_index, max_index)), 
                        yticklabels=list(range(min_index, max_index))
            )
            
            plt.title(f"Heatmap of {title} Chain A")

            if display == True:
            # Sauvegarder et afficher
            # plt.savefig(f"heatmap_combined_hbonds.png")
                plt.show()



def main():
    """Do main system"""
    # system = System()
    file_path = check_file_exists()
    system = read_pdb(file_path)
    system = remove_residues(system)

    # for atom in system.atoms:
    #     print(atom.chain, atom.resid, atom.name)


    system.hbonds_calc() 

    dssp = Dssp(system = system,
                nturn = [],
                helix = [],
                bridge = [],
                ladder = [],
                sheet = None
                )

    dssp.system = system

    dssp.nturn_calc()
    dssp.helices_calc()
    # indices = dssp.helices_calc()
    # # dssp.color_pymol(indices, color = 'red')


    dssp.bridge_calc()
    dssp.ladder_calc()
    # indices = dssp.ladder_calc()
    # # dssp.color_pymol(indices, color = 'blue')

    # dssp.sheet_calc()

    dssp.write_DSSP()




if __name__ == "__main__":
    main()
