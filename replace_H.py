import pymol
from pymol import cmd
import math

def add_h(file_path):
    """Open PDB file and add Hydrogens"""
    # Launch PyMOL
    pymol.finish_launching(['pymol', '-c'])

    # Load the PDB file
    pymol.cmd.load(f'{file_path}', 'myprotein')

    # Add the hydrogen atoms
    pymol.cmd.h_add()

    # Save the pdb file with added hydrogens
    pymol.cmd.save(f'{file_path}')

    # Quit PyMOL
    pymol.cmd.quit()


def reindex_pdb(input_filename):
    """
    Reindex the residues in the PDB file so that the first residue in each chain starts at 1
    """
    with open(input_filename, 'r') as pdb_file:
        lines = pdb_file.readlines()

    # Create a dictionary to store the first residue number for each chain
    chain_first_residue = {}

    # Pass over the lines once to identify the first residue of each chain
    for line in lines:
        if line.startswith('ATOM'):
            chain_id = line[21]
            residue_number = int(line[22:26].strip())
            if chain_id not in chain_first_residue:
                chain_first_residue[chain_id] = residue_number

    # Rewrite the PDB file with reindexed residues
    with open(input_filename, 'w') as pdb_file:
        for line in lines:
            if line.startswith('ATOM') or line.startswith('HETATM'):

                chain_id = line[21]
                residue_number = int(line[22:26].strip())

                first_residue = chain_first_residue[chain_id]
                residue_offset = first_residue - 1

                new_residue_number = residue_number - residue_offset

                new_line = line[:22] + f"{new_residue_number:4d}" + line[26:]
                pdb_file.write(new_line)

            else:
                # Write the line without modification (TER or HEADER lines)
                pdb_file.write(line)


def parse_pdb(filename):
    """
    Parse a PDB file and return a list of atoms as dictionaries.
    """
    atoms = []
    with open(filename, 'r') as pdb_file:
        for line in pdb_file:
            if line.startswith('ATOM') or line.startswith('HETATM'):
                atom = {
                    'serial': int(line[6:11].strip()),
                    'name': line[12:16].strip(),
                    'residue': line[17:20].strip(),
                    'chain': line[21],
                    'resi': int(line[22:26].strip()),
                    'x': float(line[30:38].strip()),
                    'y': float(line[38:46].strip()),
                    'z': float(line[46:54].strip())
                }
                atoms.append(atom)
    return atoms


def distance(atom1, atom2):
    """
    Calculate the Euclidean distance between two atoms.
    """
    return math.sqrt((atom1['x'] - atom2['x']) ** 2 + (atom1['y'] - atom2['y']) ** 2 + (atom1['z'] - atom2['z']) ** 2)


def find_closest_hydrogen(nitrogen_atom, atoms):
    """
    Find the closest hydrogen to a nitrogen atom and rename it to 'H'.
    """
    closest_hydrogen = None
    min_distance = float('inf')

    # Select all hydrogen atoms in the same residue
    for atom in atoms:
        if atom['residue'] == nitrogen_atom['residue'] and atom['chain'] == nitrogen_atom['chain'] and atom['name'].startswith('H'):
            dist = distance(nitrogen_atom, atom)
            if dist < min_distance and dist < 1.2:
                min_distance = dist
                closest_hydrogen = atom

    # Rename the closest hydrogen
    if closest_hydrogen:
        closest_hydrogen['name'] = 'H'

    return closest_hydrogen


def modify_pdb(atoms, output_filename):
    """
    Modify the PDB file by renaming hydrogen atoms to 'H' and save the modified file.
    """
    with open(output_filename, 'w') as pdb_file:
        for atom in atoms:
            line = f"ATOM  {atom['serial']:5d} {atom['name']:<4} {atom['residue']} {atom['chain']}{atom['resi']:4d}    {atom['x']:8.3f}{atom['y']:8.3f}{atom['z']:8.3f}  1.00  0.00\n"
            pdb_file.write(line)


def process_pdb(input_filename):
    """
    Process the PDB file by finding and renaming the closest hydrogens to nitrogen atoms.
    """
    # Add hydrogens
    add_h(input_filename)

    # Reindex residue numbers
    reindex_pdb(input_filename)

    # Parse the PDB file
    atoms = parse_pdb(input_filename)
    modify_pdb(atoms, input_filename)

    # Find all nitrogen atoms
    nitrogen_atoms = [atom for atom in atoms if atom['name'] == 'N']

    # For each nitrogen atom find the closest hydrogen and rename it
    for nitrogen_atom in nitrogen_atoms:
        find_closest_hydrogen(nitrogen_atom, atoms)

    # Save the modified file
    output_filename = f"{input_filename.split('.')[0]}_modified.pdb"
    modify_pdb(atoms, output_filename)

    return output_filename