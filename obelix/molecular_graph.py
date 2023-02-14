import numpy as np
import periodictable
from morfeus import read_xyz as read_xyz_mf
from morfeus.utils import convert_elements
from morfeus.io import read_cclib
import operator


### Temporary two solutions for storing the atomic radii.

atomic_radii = {}
for element in periodictable.elements:
    atomic_radii[element.symbol] = element.covalent_radius

atomic_radii = dict(
    Ac=1.88,
    Ag=1.59,
    Al=1.35,
    Am=1.51,
    As=1.21,
    Au=1.50,
    B=0.83,
    Ba=1.34,
    Be=0.35,
    Bi=1.54,
    Br=1.21,
    C=0.68,
    Ca=0.99,
    Cd=1.69,
    Ce=1.83,
    Cl=0.99,
    Co=1.33,
    Cr=1.35,
    Cs=1.67,
    Cu=1.52,
    D=0.23,
    Dy=1.75,
    Er=1.73,
    Eu=1.99,
    F=0.64,
    Fe=1.34,
    Ga=1.22,
    Gd=1.79,
    Ge=1.17,
    H=0.23,
    Hf=1.57,
    Hg=1.70,
    Ho=1.74,
    I=1.40,
    In=1.63,
    Ir=1.32,
    K=1.33,
    La=1.87,
    Li=0.68,
    Lu=1.72,
    Mg=1.10,
    Mn=1.35,
    Mo=1.47,
    N=0.68,
    Na=0.97,
    Nb=1.48,
    Nd=1.81,
    Ni=1.50,
    Np=1.55,
    O=0.68,
    Os=1.37,
    P=1.05,
    Pa=1.61,
    Pb=1.54,
    Pd=1.50,
    Pm=1.80,
    Po=1.68,
    Pr=1.82,
    Pt=1.50,
    Pu=1.53,
    Ra=1.90,
    Rb=1.47,
    Re=1.35,
    Rh=1.45,
    Ru=1.40,
    S=1.02,
    Sb=1.46,
    Sc=1.44,
    Se=1.22,
    Si=1.20,
    Sm=1.80,
    Sn=1.46,
    Sr=1.12,
    Ta=1.43,
    Tb=1.76,
    Tc=1.35,
    Te=1.47,
    Th=1.79,
    Ti=1.47,
    Tl=1.55,
    Tm=1.72,
    U=1.58,
    V=1.33,
    W=1.37,
    Y=1.78,
    Yb=1.94,
    Zn=1.45,
    Zr=1.56,
)


class MolGraph:
    """
    This is a molecular graph class, containing several functionalities: 
    finding the adjancecy matrix, adjancecy list. Class modified from /xyz2graph/xyz2graph.py.
    """

    __slots__ = [
        "elements",
        "x",
        "y",
        "z",
        "adj_list",
        "atomic_radii",
        "bond_lengths",
        "adj_matrix",
    ]

    def __init__(self):
        self.elements = []
        self.x = []
        self.y = []
        self.z = []
        self.adj_list = {}
        self.atomic_radii = []
        self.bond_lengths = {}
        self.adj_matrix = None

    def read_xyz_coord_from_mf(self, elements, coordinates) -> None:
        """Reads elements and coordinates from morfeus, searches for elements and their cartesian coordinates
        and adds them to corresponding arrays."""

        self.elements = elements[:]
        
        self.x = coordinates[:, 0]
        self.y = coordinates[:, 1]
        self.z = coordinates[:, 2]
        
        self.atomic_radii = [atomic_radii[element] for element in self.elements]
        self._generate_adjacency_list()

    def _generate_adjacency_list(self):
        """Generates an adjacency list from atomic cartesian coordinates."""

        node_ids = range(len(self.elements))
        xyz = np.stack((self.x, self.y, self.z), axis=-1)
        distances = xyz[:, np.newaxis, :] - xyz
        distances = np.sqrt(np.einsum("ijk,ijk->ij", distances, distances))

        atomic_radii = np.array(self.atomic_radii)
        distance_bond = (atomic_radii[:, np.newaxis] + atomic_radii) * 1.3

        adj_matrix = np.logical_and(0.1 < distances, distance_bond > distances).astype(int)

        for i, j in zip(*np.nonzero(adj_matrix)):
            self.adj_list.setdefault(i, set()).add(j)
            self.adj_list.setdefault(j, set()).add(i)
            self.bond_lengths[frozenset([i, j])] = round(distance_bond[i, j], 5)

        self.adj_matrix = adj_matrix


def bfs(visited, graph, node):
    """
    This function is a classic breadth-first traversal algorithm. It is applied to find all the subgraphs
    when the metal center is removed.

    :param visited:
    :param graph:
    :param node:
    
    :return visited: 
    """
    queue = []
    
    visited.append(node)
    queue.append(node)

    while queue:
        s = queue.pop(0)
        for neighbour in graph[s]:
            if neighbour not in visited:
                visited.append(neighbour)
                queue.append(neighbour)
                
    return visited


def molecular_graph(elements, coordinates):
    """
    This function is applied to find the molecular graph of a TM complex.
    This function returns the indices of all atoms making the investigated and the indices forming the bite angle.

    :param elements:
    :param coordinates:

    :return ligand, bidentate:
    """
    metal_centers = ['Rh', 'Ru', 'Mn', 'Pd', 
                     'Ir', 'Pt', 'Co']

    periodic_table = [element.symbol for element in periodictable.elements]
    
    ### Check if elements coming from morfeus are given as numbers.
    # If not, make conversion using the periodic table package.
    
    if str(elements[0]).isdigit():
        elements = list(elements)
        for elem_id, element in enumerate(elements):    
            elements[elem_id] = periodic_table[element]
        elements = np.array(elements)
    
    ### Call to the MolGraph() functionality 
    mg = MolGraph()
    mg.read_xyz_coord_from_mf(elements=elements,coordinates=coordinates)
    graph = mg.adj_list
    
    for elem_id, element in enumerate(elements):
        if element in metal_centers:
           metal_center_id = int(elem_id)
           break

    metal_center_bonds_ = graph[metal_center_id]
    del graph[metal_center_id]
    
    for key, bonds_to_atom in zip(graph.keys(), graph.values()):    
        graph[key] = list(bonds_to_atom)
    
        for bond_id, bond in enumerate(bonds_to_atom):
            if bond == metal_center_id:
                graph[key].pop(bond_id)
                
    store_ligands = []

    for node in metal_center_bonds_:
        store_ligands.append(list(np.sort(bfs([], graph, node))))

    # Remove duplicates and store in new list --> clean_ligands
    clean_ligands = []

    for ligand in store_ligands:
        if ligand not in clean_ligands:
            clean_ligands.append(ligand)
    
    
    # store the size of the ligands in a list
    ligand_sizes = []
    
    for ligand in clean_ligands:
        ligand_sizes.append(len(ligand))

    # This part checks whether two ligands have the max length
    index_max_ligand = np.argwhere(ligand_sizes == np.amax(ligand_sizes))
    # Flattens 2D array
    index_max_ligand = index_max_ligand.ravel()

    # Check if two monodentate ligands instead of one single bidentate ligand    
    if len(index_max_ligand) == 2:
        ligand = []
        ligand.extend(clean_ligands[index_max_ligand[0]])
        ligand.extend(clean_ligands[index_max_ligand[1]])
    else:
        ligand = clean_ligands[index_max_ligand[0]]

    # Store metal id in a list
    bidentate = [metal_center_id]
    # Extend bidentate atoms with the donating atoms
    for bond_to_metal in metal_center_bonds_:
        if bond_to_metal in ligand:
            bidentate.append(bond_to_metal)
    # Check whether more atoms are taken as donating: especially important for the pristine structures
    if len(bidentate) > 3:
        # check atom type
        new_bidentate = [metal_center_id]
        
        ### Check whether the atom is P, N or S and rewrite the bidentate list with only these types
        
        for ligating_atom in bidentate:
            if (elements[ligating_atom] == 'N') or (elements[ligating_atom] == 'P') or (elements[ligating_atom] == 'S'):
                new_bidentate.append(ligating_atom)
    
        ### Check two closest donors to the metal center.
        
        if len(new_bidentate) == 2:
            return ligand, new_bidentate
    
        else:
            dict_distances = {}
            new_bidentate = new_bidentate[1:]
            for ligating_atom in new_bidentate:
                dict_distances[ligating_atom] = np.linalg.norm(coordinates[metal_center_id, :] - coordinates[ligating_atom, :])
            
            dict_distances = {k: v for k, v in sorted(dict_distances.items())}
            
            # Sort dictionary according to bond length to the metal center
            dict_distances = dict(sorted(dict_distances.items(), key=operator.itemgetter(1),reverse=False))
            new_bidentate = []
            new_bidentate.extend([metal_center_id, list(dict_distances.keys())[0], list(dict_distances.keys())[1]])
            return ligand, new_bidentate
    return ligand, bidentate


### testing -- testing successful on xyz and log
# log = '[Rh+1]_L98_SP0.log'
# elements, coordinates = read_cclib(log)
# molecular_graph(elements = elements, coordinates = coordinates)

# xyz = '1441830-74-5_NBD.xyz'
# elements, coordinates = read_xyz_mf(xyz)
# molecular_graph(elements = elements, coordinates = coordinates)
