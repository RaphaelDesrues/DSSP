def find_incomplete_residues(pdb_filename):
    """
    Lit un fichier PDB et retourne une liste des résidus qui ne possèdent pas 
    la combinaison des atomes C, CA, N.
    """
    # Dictionnaire pour suivre les atomes présents dans chaque résidu
    residues = {}

    # Lire le fichier PDB
    with open(pdb_filename, 'r') as pdb_file:
        for line in pdb_file:
            # Sélectionner les lignes contenant des atomes (ATOM ou HETATM)
            if line.startswith('ATOM') or line.startswith('HETATM'):
                # Extraire le résidu (nom de la chaîne et le numéro du résidu)
                chain_id = line[21]  # Colonne 21 pour la chaîne
                residue_number = int(line[22:26].strip())  # Colonne 22-26 pour le numéro du résidu
                atom_name = line[12:16].strip()  # Colonne 12-16 pour le nom de l'atome
                
                # Créer une clé unique pour chaque résidu (chaîne + numéro)
                residue_key = (chain_id, residue_number)

                # Ajouter l'atome au dictionnaire du résidu
                if residue_key not in residues:
                    residues[residue_key] = set()
                residues[residue_key].add(atom_name)

    # Liste des résidus incomplets (qui manquent l'une des atomes C, CA ou N)
    incomplete_residues = []

    for residue_key, atoms in residues.items():
        if not {'C', 'CA', 'N'}.issubset(atoms):
            incomplete_residues.append(residue_key)

    return incomplete_residues


# Exemple d'utilisation
pdb_filename = "1g5j.pdb"  # Remplacer par le nom de votre fichier PDB
incomplete_residues = find_incomplete_residues(pdb_filename)

print("Résidus incomplets (sans C, CA, N) :")
for chain_id, residue_number in incomplete_residues:
    print(f"Chaîne {chain_id}, Résidu {residue_number}")


