"""
Protein Secondary Structure Prediction using the DSSP method
============================================================
Description :
    This script predicts the secondary structures of proteins using DSSP-like algorithms
    - Reference :
    Kabsch, W., et C. Sander. « Dictionary of Protein Secondary Structure: Pattern Recognition of Hydrogen-Bonded and Geometrical Features ».
    Biopolymers 22, no 12 (décembre 1983): 2577 2637. https://doi.org/10.1002/bip.360221211.
Usage :
    python main.py <PDB_FILE>
Author :
    Raphaël DESRUES - M2 ISDD
"""


import os
import sys
import string
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # ou 'Agg', 'Qt5Agg', 'TkAgg' etc.
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# pymol package to add Hydrogens
# Self-made package to correctly name hydrogens
import pymol
from pymol import cmd
import replace_H


# Constantes
Q1 = 0.42
Q2 = 0.20
DIM_F = 332
ENERGY_CUTOFF = 0.5
ABT = string.ascii_uppercase * 4


def check_file_exists():
    """Checks if the given pdb file exists

    Returns
    -------
    file_path : str
        Path to the PDB file if it exists
    """
    if len(sys.argv) != 2:
        sys.exit("Il faut un fichier PDB")

    file_path = sys.argv[1]

    return file_path


def read_pdb(file_path):
    """Read a PDB file and extract atoms from the protein chains

    Parameters
    ----------
    file_path : str
        The path to the PDB file

    Returns
    -------
    system : class
        The class with the parsed PDB
    """
    atoms = ["C", "O", "N", "H", "CA"]

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

                system.add_atom(chain, resid, resname, name, position)

    return system


def remove_residues(system):
    """Remove the incomplete residues.

    Incomplete residues are those missing either C, O, N, H or CA

    Parameters
    ----------
    system : class
        Instance of the class System

    Returns
    -------
    system : class
        Instance of the updated class System
    """
    chains = {atom.chain for atom in system.atoms}
    chains = sorted(chains)

    for chain in chains:

        residue_count = {}

        atoms = [atom for atom in system.atoms if atom.chain == chain]

        for atom in atoms:
            if atom.resid in residue_count:
                residue_count[atom.resid] += 1

            else:
                residue_count[atom.resid] = 1

        system.atoms.extend([atom for atom in atoms if residue_count[atom.resid] == 5 or atom.resname == "PRO"])

    return system


class Atom:
    """Protein's atoms"""

    def __init__(
        self,
        chain: str,
        resid: int,
        resname: str,
        name: str,
        position: np.ndarray
    ):
        """Initialize an Atom object

        Parameters
        ==========
        chain : str
            Chain ID of the atom
        resid : int
            Residue ID of the atom
        resname : str
            Name of the residue of the atom
        name : str
            Name of the atom
        position : np.ndarray
            3D coordinates of the atom
        """

        self.chain = chain
        self.resid = resid
        self.resname = resname
        self.name = name
        self.position = position


class System:
    """Protein's topology and H-bonds"""

    def __init__(
        self,
        atoms: list,
        hbonds: list[np.ndarray]
    ):
        """Initialize a System object

        Each chain are considered separated

        Parameters
        ==========
        atoms : list
            Atoms's list (Object Atom) of the protein
        hbonds : list[np.ndarray]
            H-bonds matrix of the protein
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
        resname: str,
        name: str,
        position: np.ndarray
    ):
        """Add an atom to the class System

        Parameters
        ----------
        chain : str
            Chain ID of the atom
        resid : str
            Residue ID of the atom
        resname : str
            Residue name of the atom
        name : str
            Atom name
        position : np.ndarray
            3D coordinates of the atom
        """
        atom = Atom(chain, resid, resname, name, position)
        self.atoms.append(atom)


    def dist(self, atom1: np.ndarray, atom2: np.ndarray) -> float:
        """Compute distance between two atoms

        Parameters
        ----------
        atom1 : np.ndarray
            Coordinates of the first atom
        atom2 : np.ndarray
            Coordinates of the second atom

        Returns
        -------
        d : float
            Distance between the two atoms
        """
        atom1 = np.array(atom1)
        atom2 = np.array(atom2)

        d = np.sqrt(((atom1 - atom2)**2).sum())

        return d


    def angle(self, atom1: np.ndarray, atom2: np.ndarray, atom3: np.ndarray) -> float:
        """Compute angle formed by three atoms

        Parameters
        ----------
        atom1 : np.ndarray
            Coordinates of the first atom
        atom2 : np.ndarray
            Coordinates of the second atom
        atom3 : np.ndarray
            Coordinates of the third atom

        Returns
        -------
        angle : float
            Angle in degrees
        """
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


    def dihedral_angle(self, atom1: np.ndarray, atom2: np.ndarray, atom3: np.ndarray, atom4: np.ndarray) -> float:
        """Compute dihedral angles formed by four atoms

        Parameters
        ----------
        atom1 : np.ndarray
            Coordinates of the first atom
        atom2 : np.ndarray
            Coordinates of the second atom
        atom3 : np.ndarray
            Coordinates of the third atom
        atom4 : np.ndarray
            Coordinates of the fourth atom

        Returns
        -------
        angle_diedre : float
            Diedre angle in degrees
        """
        atom1 = np.array(atom1)
        atom2 = np.array(atom2)
        atom3 = np.array(atom3)
        atom4 = np.array(atom4)

        # Vectors of the edges of each plane
        v1 = (atom2 - atom1).flatten()
        v2 = (atom3 - atom2).flatten()
        v3 = (atom3 - atom2).flatten()
        v4 = (atom4 - atom3).flatten()

        # Normal vectors to the planes ABC et BCD
        n1 = np.cross(v1, v2)
        n2 = np.cross(v3, v4)

        n1_norm = np.linalg.norm(n1)
        n2_norm = np.linalg.norm(n2)

        scalar_product = np.dot(n1, n2)

        angle_diedre = np.arccos(scalar_product / (n1_norm * n2_norm))

        angle_diedre = np.degrees(angle_diedre)

        return angle_diedre


    def energy_hbond(self, atom1, atom2, atom3, atom4):
        """Compute energy in a given H-bonding group

        Parameters
        ----------
        atom1 : np.ndarray
            Coordinates of the first atom
        atom2 : np.ndarray
            Coordinates of the second atom
        atom3 : np.ndarray
            Coordinates of the third atom
        atom4 : np.ndarray
            Coordinates of the fourth atom

        Returns
        -------
        energy : float
            Energy of the given hbond
        """

        dist_ON = 1 / self.dist(atom1, atom2)
        dist_CH = 1 / self.dist(atom3, atom4)
        dist_OH = 1 / self.dist(atom2, atom4)
        dist_CN = 1 / self.dist(atom1, atom3)

        energy = Q1 * Q2 * DIM_F * (dist_ON + dist_CH - dist_OH - dist_CN)

        return energy


    def hbonds_calc(self):
        """Compute hbonds in whole protein

        Compute hydrogen bonds based on distance, angle and energy criteria
        and store the results in a boolean matrix for each chain
        """
        chains = {atom.chain for atom in self.atoms}
        chains = sorted(chains)

        self.hbonds = {}

        for chain in chains:

            N_atoms = [atom1 for atom1 in self.atoms if atom1.name == "N" and atom1.chain == chain]
            O_atoms = [atom2 for atom2 in self.atoms if atom2.name == "O" and atom2.chain == chain]
            H_atoms = [atom3 for atom3 in self.atoms if atom3.name == "H" and atom3.chain == chain]
            C_atoms = [atom4 for atom4 in self.atoms if atom4.name == "C" and atom4.chain == chain]

            max_resid = max([atome.resid for atome in self.atoms if atome.chain == chain]) + 6 # +6 car test i/j + 5 dans nturn_calc (hbond_temp)
            chain_index = (np.zeros((max_resid, max_resid), dtype=bool))

            for atom1, atom3 in zip(N_atoms, H_atoms):

                for atom2, atom4 in zip(O_atoms, C_atoms):

                    # Compute O---N distance
                    dist = self.dist(atom1.position, atom2.position)

                    # Compute O---H-N angle
                    angle = self.angle(atom1.position, atom2.position, atom3.position)

                    if dist < 5.2 and angle < 63:

                        energy = self.energy_hbond(atom1.position, atom2.position, atom3.position, atom4.position)

                        if energy < ENERGY_CUTOFF:

                            chain_index[atom1.resid][atom2.resid] = True

                        else:
                            continue

            self.hbonds[chain] = chain_index

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

            plt.title(f"Heatmap of hbonds Chain A")

            # Sauvegarder et afficher
            # plt.savefig(f"heatmap_combined_hbonds.png")
            plt.show()


class Dssp:
    """Compute DSSP : Define Secondary-Structure of Protein"""

    def __init__(
        self,
        system,
        nturn: list,
        helix: list,
        bridge: dict,
        ladder: dict,
        sheet: list,
        bend: dict,
        chirality: dict
    ):
        """Initialize a DSSP object

        Parameters
        ==========
        system : System
            Protein system containing atoms and hydrogen bonds
        nturn : list
            List of n-turns
        helix : list
            List of helices
        bridge : dict
            Dictionary of bridges
        ladder : dict
            Dictionary of ladders
        sheet : list
            list of sheets
        bend : dict
            Dictionary of bends
        chirality : dict
            Dictionary of chiralities
        """

        self.system = system
        self.nturn = nturn
        self.helix = helix
        self.bridge = bridge
        self.ladder = ladder
        self.sheet = sheet
        self.bend = bend
        self.chirality = chirality


    def nturn_calc(self):
        """Compute nturn (3-turns, 4-turns and 5turns) based on hbonds

        Results
        -------
        self.nturn : dict
            Dictionnary with nturn matrix for each chain

        Notes
        -----
        Hydrogen bonds are retrieved from the self.system.hbonds attribute
        n-turns can be visualized with heatmap
        """

        chains = {atom.chain for atom in self.system.atoms}
        chains = sorted(chains)

        self.nturn = {}

        for chain in chains:

            hbonds_temp = self.system.hbonds[chain]

            max_resid = max([atome.resid for atome in self.system.atoms if atome.chain == chain]) + 1

            self.nturn[chain] = np.zeros((max_resid, max_resid), dtype = int)

            for i in range(len(hbonds_temp)):

                for j in range(i):

                    if hbonds_temp[i][j] == True:

                        if (hbonds_temp[i + 3][j] == True) or hbonds_temp[i][j + 3] == True:

                            self.nturn[chain][i][j] = 3


                        if (hbonds_temp[i + 4][j] == True) or hbonds_temp[i][j + 4] == True:

                            self.nturn[chain][i][j] = 4


                        if (hbonds_temp[i + 5][j] == True) or hbonds_temp[i][j + 5] == True:

                            self.nturn[chain][i][j] = 5

                        else:
                            continue

        self.plot_heatmap(matrix = self.nturn, title = "nturn", display = False)


    def helices_calc(self):
        """Compute alpha-helices in the protein structure

        Helices are identified thanks to 4-turn patterns
        2 consecutive 4-turn residues are part of a alpha-helix

        Results
        -------
        self.helix : dict
            Dictionnary with helices matrix for each chain

        Notes
        -----
        Helices can be visualized with heatmap
        """

        chains = {atom.chain for atom in self.system.atoms}
        chains = sorted(chains)

        self.helix = {}

        for chain in chains:

            max_resid = max([atome.resid for atome in self.system.atoms if atome.chain == chain]) + 1
            self.helix[chain] = np.zeros((max_resid, max_resid), dtype = int)

            for i in range(len(self.nturn[chain])):

                for j in range(i):

                    if self.nturn[chain][i][j] == 4 and self.nturn[chain][i + 1][j + 1] == 4:

                        self.helix[chain][i, j] = 1
                        self.helix[chain][i + 1][j + 1] = 1
                        break

        self.plot_heatmap(self.helix, title = "Helix", display = False)

        for chain in chains:

            p = 0
            mat_tmp = self.helix[chain]

            max_resid = max([atome.resid for atome in self.system.atoms if atome.chain == chain])
            self.helix[chain] = np.zeros((max_resid, max_resid), dtype = object)

            for i in range(len(mat_tmp)):

                for j in range(i):

                    if mat_tmp[i, j] != 0 and mat_tmp[i][j] == mat_tmp[i + 1][j + 1]:

                        self.helix[chain][i, j] = ABT[p]
                        self.helix[chain][i + 1, j + 1] = ABT[p]
                        break

                    elif mat_tmp[i, j] != 0 and mat_tmp[i][j] != mat_tmp[i + 1][j + 1]:
                        self.helix[chain][i, j] = ABT[p]
                        p += 1



    def bridge_calc(self):
        """Compute bridges in a protein structure from hbonds

        Bridges are formed between residues pairs with specific hydrogen bonds arrangments
        Parallel : Hbond(i - 1, j) and Hbond(j, i + 1) or Hbond(j - 1, i) and Hbond(i, j + 1)
        Antiparallel : Hbond(i, j) and Hbond(j, i) or Hbond(i - 1, j) and Hbond(j - 1, i + 1)

        Results
        -------
        self.bridge : dict
            Dictionnary with bridge matrix for each chain
            A : indicate parallel bridges
            AP : indicate antiparallel bridges

        Notes
        -----
        Helices can be visualized with heatmap
        """
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

            max_resid = max([atome.resid for atome in self.system.atoms if atome.chain == chain])
            self.bridge[chain] = np.zeros((max_resid, max_resid))

            for i in range(len(resid_list) - 2):

                for j in range(i):

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
        """Compute beta-ladder in the protein

        Beta-ladders are formed when residues in parallel and anti-parallel bridges are connected

        Results
        -------
        self.ladder : dict
            Dictionnary with ladder matrix for each chain

        Notes
        -----
        Helices can be visualized with heatmap
        """

        chains = {atom.chain for atom in self.system.atoms}
        chains = sorted(chains)

        self.ladder = {}

        for chain in chains:

            max_resid = max([atome.resid for atome in self.system.atoms if atome.chain == chain])
            self.ladder[chain] = np.zeros((max_resid, max_resid), dtype = float)

            for i in range(1, len(self.bridge[chain]) - 1):

                for j in range(1, len(self.bridge[chain]) - 1):

                    if self.bridge[chain][i, j] != 0 and self.bridge[chain][i, j] == self.bridge[chain][i - 1, j + 1]:
                        self.ladder[chain][i, j] = self.bridge[chain][i, j]
                        self.ladder[chain][i - 1, j + 1] = self.bridge[chain][i - 1, j + 1]

                    else:
                        continue

        self.plot_heatmap(self.ladder, title = "Ladder", display = False)


    def sheet_calc(self):
        """Compute beta-sheet in the protein

        Beta-sheets are groups of beta-ladders that are connected in the sequence
        This function assigns unique labels to each sheet for each chain

        Results
        -------
        self.sheet : dict
            Dictionnary with sheet matrix for each chain

        Notes
        -----
        Sheet can be visualized with heatmap - Need letter transformation beforehand
        """

        chains = {atom.chain for atom in self.system.atoms}
        chains = sorted(chains)

        self.sheet = {}
        ind_ladder = {}
        ind_temp = []

        for chain in chains:

            max_resid = max([atome.resid for atome in self.system.atoms if atome.chain == chain])
            self.sheet[chain] = np.zeros((max_resid, max_resid), dtype = object)
            ind_ladder[chain] = []

            for i in range(1, len(self.ladder[chain]) - 2):

                for j in range(i):
                    # print("COND1", "i", i, "j", j, self.ladder[chain][i, j])
                    # print("COND2", "i", i, "j", j, self.ladder[chain][i + 1, j - 1])

                    if self.ladder[chain][i, j] != 0 and self.ladder[chain][i + 1, j - 1] != 0:
                        # print("TRUE")
                        ind_temp.append([i, j])
                        ind_temp.append([i + 1, j - 1])

                    else:
                        if ind_temp == []:
                            continue

                        else:
                            ind_temp = [ind for sub_ind in ind_temp for ind in sub_ind]
                            # print("GOOD", ind_temp)
                            ind_ladder[chain].append(list(dict.fromkeys(ind_temp)))
                            ind_temp = []
                            break
            # print(ind_ladder[chain])

            abt_count = 0

            for m in range(len(ind_ladder[chain])):

                i_temp = ind_ladder[chain][m][0]
                j_temp = ind_ladder[chain][m][1]
                i_temp_2 = ind_ladder[chain][m][2]
                j_temp_2 = ind_ladder[chain][m][3]

                if m == 0:
                    self.sheet[chain][i_temp, j_temp] = ABT[abt_count]
                    self.sheet[chain][i_temp_2, j_temp_2] = ABT[abt_count]
                    continue

                if any(elem in ind_ladder[chain][m] for elem in ind_ladder[chain][m - 1]):
                    self.sheet[chain][i_temp, j_temp] = ABT[abt_count]
                    self.sheet[chain][i_temp_2, j_temp_2] = ABT[abt_count]

                else:
                    abt_count += 1
                    self.sheet[chain][i_temp, j_temp] = ABT[abt_count]
                    self.sheet[chain][i_temp_2, j_temp_2] = ABT[abt_count]

        # self.plot_heatmap(self.sheet, title = "Sheet", display = True) # Fonctionne pas si Lettres


    def bend_calc(self):
        """Compute bend in the protein

        Bend are regions where 5 consecutive residues deviate in their backbone angle of more than 70 degrees

        Results
        -------
        self.bend : dict
            Dictionary of list of bending residues for each chain
        """

        chains = {atom.chain for atom in self.system.atoms}
        chains = sorted(chains)

        for chain in chains:

            self.bend[chain] = {}

            c_alpha = [atom for atom in self.system.atoms if atom.name == "CA" and atom.chain == chain]

            for i in range(2, len(c_alpha) - 3):

                angle = self.system.angle(c_alpha[i - 2].position, c_alpha[i].position, c_alpha[i + 2].position) # faisable ?

                if angle > 70:
                    for j in range(i - 2, i + 3):
                        self.bend[chain][c_alpha[j].resid] = 'S'

            # print('BEND', self.bend[chain])


    def chirality_calc(self):
        """Compute chirality in the protein

        Chirality is calculated using dihedral angles between alpha-carbons
        Residues have either '+' or '-' chirality

        Results
        -------
        self.chirality : dict
            Dictionary of list of residues' chirality for each chain
        """

        chains = {atom.chain for atom in self.system.atoms}
        chains = sorted(chains)

        for chain in chains:

            self.chirality[chain] = {}

            c_alpha = [atom for atom in self.system.atoms if atom.name == "CA" and atom.chain == chain]

            for i in range(1, len(c_alpha) -3):

                angle_diedre = self.system.dihedral_angle(c_alpha[i - 1].position, c_alpha[i].position, c_alpha[i + 1].position, c_alpha[i + 3].position)

                # print('ANGLE DIEDRE', angle_diedre)

                if angle_diedre > 0 and angle_diedre < 180:

                    self.chirality[chain][c_alpha[i].resid] = '+'

                else:

                    self.chirality[chain][c_alpha[i].resid] = '-'


            # print('CHIRALITY', self.chirality[chain])


    def write_DSSP(self):
        """Write a DSSP file

        Convert previous protein properties into a pandas df
        The panda df is then convert into a text file

        Results
        -------
        df_dssp : text file
            Output the DSSP in a text file
        """
        # Create the df for dssp output
        df_dssp = pd.DataFrame(columns=['Chain', 'Sequence', 'Name', 'Sheet', 'Bridge', 'Chirality', 'Bend', 'Helix', '5-turn', '4-turn', '3-turn', 'Summary'])

        chains = {atom.chain for atom in self.system.atoms}
        chains = sorted(chains)

        for chain in chains:

            # Create a temporary dssp df for each chain
            df = pd.DataFrame(columns=['Chain', 'Sequence', 'Name', 'Sheet', 'Bridge', 'Chirality', 'Bend', 'Helix', '5-turn', '4-turn', '3-turn', 'Summary'])

            # Add the residues ID
            resid_values = [atome.resid for atome in self.system.atoms if atome.chain == chain]
            df['Sequence'] = np.arange(min(resid_values), max(resid_values) + 1, dtype = int)

            # Add the chain name
            df['Chain'] = chain

            previous_resid = min(resid_values) - 1
            for atome in self.system.atoms:
                if atome.chain == chain and atome.resid != previous_resid:
                    df.loc[atome.resid - 1, 'Name'] = atome.resname
                    previous_resid = atome.resid

            # Initialize all the columns with empty entries
            df['Sheet'] = ''
            df['Bridge'] = ''
            df['Helix'] = ''
            df['Chirality'] = ''
            df['Bend'] = ''

            # Fill using the lists
            df['Chirality'] = df['Sequence'].apply(lambda x: self.chirality[chain].get(x, None))
            df['Bend'] = df['Sequence'].apply(lambda x: self.bend[chain].get(x, None))

            # Fill using the dictionaries
            for i in range(max(resid_values)):
                df.loc[i - 1, 'Sheet'] = next((x for x in self.sheet[chain][i] if x != 0), '')
                df.loc[i - 1, 'Bridge'] = next((x for x in self.ladder[chain][i] if x != 0), '')
                df.loc[i - 1, 'Helix'] = next((x for x in self.helix[chain][i] if x != 0), '')
                df.loc[i - 1, '5-turn'] = next((x for x in self.nturn[chain][i] if x == 5), '')
                df.loc[i - 1, '4-turn'] = next((x for x in self.nturn[chain][i] if x == 4), '')
                df.loc[i - 1, '3-turn'] = next((x for x in self.nturn[chain][i] if x == 3), '')

            # SUMMARY
            # Fill the summary according the previous columns
            for i, row in df.iterrows():
                if row['4-turn'] == 4.0:
                    df.at[i, 'Summary'] = 'H'
                elif row['3-turn'] == 3.0:
                    df.at[i, 'Summary'] = 'G'
                elif row['5-turn'] == 5.0:
                    df.at[i, 'Summary'] = 'I'
                else:
                    df.at[i, 'Summary'] = ''

                if row['Bridge'] != '':
                    df.at[i, 'Summary'] = 'E'
                if row['Bridge'] == 2.0:
                    df.at[i, 'Bridge'] = 'AP'

                if row['Bridge'] == 1.0:
                    df.at[i, 'Bridge'] = 'P'

            # Concat the chain df into the final dssp df
            df_dssp = pd.concat([df_dssp, df])


        # Save the dssp to .txt file
        df_dssp.to_csv('df_dssp.txt', sep='\t', index=False, header=False)

        # Modify the header to adjust their spaces
        with open('df_dssp.txt', 'r') as f:
            content = f.read()

        title = f"{sys.argv[1].split('.')[0]}\n"
        header = 'Chain Sequence  Name  Sheet  Bridge Chirality Bend   Helix  5-turn   4-turn  3-turn  Summary\n'

        # Save the dssp to .txt file
        with open('df_dssp.txt', 'w') as f:
            f.write(title + header + content)

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
            plt.clf()
            sns.heatmap(subset_matrix, cmap='coolwarm',  annot=False,
                        xticklabels=list(range(min_index, max_index)),
                        yticklabels=list(range(min_index, max_index))
            )



            if display == True:
                # Sauvegarder et afficher
                plt.title(f"Heatmap of {title} Chain A")
                # plt.savefig(f"heatmap_combined_hbonds.png")
                plt.show()


def main():
    """Execute the DSSP workflow

    It executes the following steps:
    1. Checks if the PDB file exists
    2. Adjusts H-N name with 'H' for consistency
    3. Reads and parses the PDB file to create a 'System' object
    4. Cleans the system by removing the incomplete residues for consistency
    5. Computes hydrogen bonds in the protein structure
    6. Initializes a 'DSSP' object to perform secondary structure analysis
    7. Executes secondary structure calculations:
        - n-turns
        - alpha-helices
        - beta-bridges
        - beta-ladders
        - beta-sheets
        - bends
        - chirality
    8. Writes the DSSP analysis results to a .txt file

    Notes
    -----
    The correct DSSP analysis depend on a correctly PDB file provided
    """
    # system = System()

    file_path = check_file_exists()

    # Optional : replace N-H names with 'H'
    file_path = replace_H.process_pdb(file_path)

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
                sheet = None,
                bend = {},
                chirality = {}
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

    dssp.sheet_calc()

    dssp.bend_calc()
    dssp.chirality_calc()

    dssp.write_DSSP()




if __name__ == "__main__":
    main()
