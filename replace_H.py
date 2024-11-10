import pymol
from pymol import cmd
import math

def add_h(file_path):
    """Open PDB file and add Hydrogens"""
    # Lancer PyMOL
    pymol.finish_launching(['pymol', '-c'])

    # Charger le fichier PDB 1g5j
    pymol.cmd.load(f'{file_path}', 'myprotein')

    # Ajouter les atomes d'hydrogène
    pymol.cmd.h_add()

    # Sauvegarder le fichier modifié avec les hydrogènes
    pymol.cmd.save(f'{file_path}')

    # Quitter PyMOL (important pour fermer proprement la session)
    pymol.cmd.quit()


def parse_pdb(filename):
    """
    Parse un fichier PDB et retourne une liste d'atomes sous forme de dictionnaires.
    """
    atoms = []
    with open(filename, 'r') as pdb_file:
        for line in pdb_file:
            if line.startswith('ATOM') or line.startswith('HETATM'):
                atom = {
                    'serial': int(line[6:11].strip()),  # Numéro atomique
                    'name': line[12:16].strip(),       # Nom de l'atome (H, N, etc.)
                    'residue': line[17:20].strip(),    # Nom du résidu
                    'chain': line[21],                 # Chaîne
                    'resi': int(line[22:26].strip()),  # Numéro de résidu
                    'x': float(line[30:38].strip()),   # Coordonnée X
                    'y': float(line[38:46].strip()),   # Coordonnée Y
                    'z': float(line[46:54].strip())    # Coordonnée Z
                }
                atoms.append(atom)
    return atoms


def distance(atom1, atom2):
    """
    Calcul la distance euclidienne entre deux atomes.
    """
    return math.sqrt((atom1['x'] - atom2['x']) ** 2 + (atom1['y'] - atom2['y']) ** 2 + (atom1['z'] - atom2['z']) ** 2)


def find_closest_hydrogen(nitrogen_atom, atoms):
    """
    Trouve l'hydrogène le plus proche d'un atome d'azote et le renomme en 'H'.
    """
    closest_hydrogen = None
    min_distance = float('inf')
    
    # Sélectionner tous les atomes d'hydrogène du même résidu
    for atom in atoms:
        if atom['residue'] == nitrogen_atom['residue'] and atom['chain'] == nitrogen_atom['chain'] and atom['name'].startswith('H'):
            dist = distance(nitrogen_atom, atom)
            if dist < min_distance and dist < 1.2:
                min_distance = dist
                closest_hydrogen = atom

    # Renommer l'hydrogène le plus proche
    if closest_hydrogen:
        closest_hydrogen['name'] = 'H'
    
    return closest_hydrogen


def modify_pdb(atoms, output_filename):
    """
    Modifie le fichier PDB en remplaçant le nom de l'hydrogène par 'H' et sauvegarde le fichier modifié.
    """
    with open(output_filename, 'w') as pdb_file:
        for atom in atoms:
            line = f"ATOM  {atom['serial']:5d} {atom['name']:<4} {atom['residue']} {atom['chain']}{atom['resi']:4d}    {atom['x']:8.3f}{atom['y']:8.3f}{atom['z']:8.3f}  1.00  0.00\n"
            pdb_file.write(line)


def process_pdb(input_filename):
    """
    Traite le fichier PDB en trouvant et en renommant les hydrogènes les plus proches des atomes d'azote.
    """

    # Ajouter les hydrogènes
    add_h(input_filename)

    # Parser le fichier PDB
    atoms = parse_pdb(input_filename)

    # Trouver tous les atomes d'azote
    nitrogen_atoms = [atom for atom in atoms if atom['name'] == 'N']
    
    # Pour chaque azote, trouver l'hydrogène le plus proche et le renommer
    for nitrogen_atom in nitrogen_atoms:
        find_closest_hydrogen(nitrogen_atom, atoms)
    
    # Sauvegarder le fichier modifié
    modify_pdb(atoms, input_filename)
