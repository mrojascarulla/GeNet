import csv
import os
import _pickle as pickle
import numpy as np

class TaxoTree(object):
    """
    Implements tree class storing a taxonomical tree.

    Stored taxonomical information to reconstruct labels and paths
    for specific microbes/genomes.

    Attributes:
        root: pointer to the root node in the tree (dummy node equal to 1).
        node_to_parent: dictionary, taxo id --> taxo id of parent in tree.
        node_to_children: dictionary, taxo id --> list of ids of children.
        node_to_group: dictionary, taxo id --> corresponding taxonomical class.
        node_to_label: dictionary, taxo_id --> label in [1, num_classes].

    Methods:
        trim_to_dataset: given the path to a csv with name of microbes,
            trim the tree and build a dictionary with ids to paths.
        id_to_label: taxo_id --> label between [1,num_classes].
        id_to_path: taxo_id --> path from root.
    """

    def __init__(self, taxo_path):
        self.node_to_parent = {}
        self.node_to_group = {}
        self.node_to_children = {}
        self.node_to_label = {}

        self.tree = None
        self.dict_id_to_path = None
        self.name_to_id = None

        # Open taxonomy file, populate node_to_parent and node_to_group 
        with open(taxo_path, 'r') as f:
            taxo = f.readlines()

        for line in taxo:
            line = line.split('|')
            taxo = int(line[0].strip('\t'))
            parent = int(line[1].strip('\t'))
            group = line[2].strip('\t')
            self.node_to_parent[taxo] = parent
            self.node_to_group[taxo] = group

        # Build dictionary node --> list of children
        self.parent_to_children = {}
        for k in self.node_to_parent:
            # No parents to root
            if k == 1:
                continue
            parent = self.node_to_parent[k]
            if parent in self.parent_to_children:
                self.parent_to_children[parent].append(k)
            else:
                self.parent_to_children[parent] = [k]

        # Build tree using self.node_to_children
        self.build_tree()
    
    def get_path_from_node(self, taxo_id):
        path = [taxo_id]
        while taxo_id != 1:
            taxo_id = self.node_to_parent[taxo_id]
            path.append(taxo_id)
        return path[::-1]

    def trim_to_dataset(self, dataset_path):
        """
        Remove all paths in the tree EXCEPT those containing elements
        from the dataset.

        Populates self.dict_id_to_path: from taxonomical id in dataset,
        returns absolute and reparametrized path from the root to
        the corresponding node.
        """
        ids_in_dataset = {}
        self.name_to_id = {}
        with open(dataset_path, 'r') as f:
            dataset = csv.reader(f)
            for row in dataset:
                #microbe_name = row[0].split(';')[0].split('.')[0]
                microbe_name = row[0].split(';')[0]
                microbe_taxo = int(row[0].split(';')[1])
                self.name_to_id[microbe_name] = microbe_taxo
                if microbe_taxo not in ids_in_dataset:
                    ids_in_dataset[microbe_taxo] = 1
        
        # Trim tree
        self.tree.trim(ids_in_dataset)
        ids_in_dataset_leaves = {}
        label = 0
        id_to_path = {}

        #Get all paths (absolute and reparametrized)
        paths, paths_reparam = self.tree.get_all_paths()
        for p in paths:
            ids_in_dataset_leaves[p[-1]] = 1

        self.ids_in_dataset = ids_in_dataset_leaves

        for i in range(len(paths)):
            for j in range(len(paths[i])):
                if (paths[i][j] in ids_in_dataset_leaves and
                    paths[i][j] not in id_to_path):

                    path_to_id = paths[i][0:j+1]
                    path_reparam_to_id = paths_reparam[i][0:j+1]
                    id_to_path[paths[i][j]] = [path_to_id, path_reparam_to_id]
                    self.node_to_label[paths[i][j]] = label
                    label += 1

        self.num_labels = len(self.node_to_label)
        self.id_to_path = id_to_path

    def load_genomes(self, genomes_path=None):
        """
        Populate self.genomes and self.paths, by loading all the genomes from 
        pickle files in genomes_path. These lists are *ordered*, so that 
        self.genomes[i] is the genome corresponding to *label* i. 
        """

        self.genomes = self.num_labels * [None]
        self.coverage = self.num_labels * [1e-10]
        self.paths = self.num_labels * [None]
        self.paths_full = self.num_labels * [None]
        self.genus_species = self.num_labels * [None]

        for k in self.name_to_id:
            taxo_id = self.name_to_id[k]
            if not taxo_id in self.node_to_label:
                continue

            label = self.node_to_label[taxo_id]
            if genomes_path:
                name_load = k.split('.')[0]
                with open(os.path.join(genomes_path, 
                                       name_load + '.pkl'), 
                          'rb') as genome:
                    self.genomes[label] = pickle.load(genome)

            current_path = self.id_to_path[taxo_id][1]
            current_path_full = self.id_to_path[taxo_id][0]
            current_path_full += (15 - len(current_path_full)) * [0]
            current_path = current_path + (15 - len(current_path)) * [0]
            self.paths[label] = current_path
            self.paths_full[label] = current_path_full 
       
        # Tools for level predictions. 
        genus_taxo_to_id = {}
        species_taxo_to_id = {}
        current_genus, current_species = 0, 0
        current_level = {}
        group_taxo_to_id = {}
        groups = ['phylum', 'class', 'order', 'family', 'genus', 'species']

        for g in groups:
            current_level[g] = 0
            group_taxo_to_id[g] = {}

        for i, p in enumerate(self.paths_full):
            p_group = [self.node_to_group[tid] for tid in p if tid != 0]
            
            self.genus_species[i] = []
            for tree_level in groups:
                if tree_level in p_group:
                    gidx = p[p_group.index(tree_level)]
                else:
                    #unknown
                    gidx = -1
                
                if gidx not in group_taxo_to_id[tree_level]:
                    group_taxo_to_id[tree_level][gidx] = current_level[tree_level]
                    current_level[tree_level] += 1   

                self.genus_species[i].append(group_taxo_to_id[tree_level][gidx])
            self.genus_species[i].append(i)
        

        self.groups = groups
        self.num_groups = {}
        self.proportions = []
        self.group_taxo_to_id = group_taxo_to_id
        for tree_level in groups:
            self.num_groups[tree_level] = len(group_taxo_to_id[tree_level].values())
            self.proportions.append(np.zeros(self.num_groups[tree_level]))

        for c in range(self.num_labels):
            for g in range(len(groups)):
                self.proportions[g][self.genus_species[c][g]] += 1

        for i in range(len(groups)):
            self.proportions[i] = [1. / p for p in self.proportions[i]]
            s = sum(self.proportions[i])
            self.proportions[i] = [p / s for p in self.proportions[i]]

    def build_tree(self):
        def build_tree_rec(tree):
            # Test if tree has children
            if tree.data in self.parent_to_children:
                children = self.parent_to_children[tree.data]
                for c in children:
                    child = Node(c)
                    build_tree_rec(child)
                    tree.add_child(child)

        # Root has taxo 1
        self.tree = Node(1)
        build_tree_rec(self.tree)

class Node(object):

    """
    Standard tree node implementation.

    Attributes:
        data: int in in node.
        children: list of Tree objects, init at None.
    """

    def __init__(self, d=None):
        self.data = d
        self.children = []

    def add_child(self, c):
        self.children.append(c)

    def get_paths_rec(self, rec, rec_reparam, l, l_reparam):
        if len(self.children) == 0:
            # We hit the leaf, rolling back
            return (rec + [self.data], rec_reparam + [0])
        for i, c in enumerate(self.children):
            new_paths = c.get_paths_rec(rec + [self.data],
                                        rec_reparam + [i],
                                        l, l_reparam)
            if new_paths is not None:
                l.append(new_paths[0])
                l_reparam.append(new_paths[1])

    def get_all_paths(self):
        """
        Returns two list with all paths in the tree from root to leaves:
        paths:          nodes are indexed with the data stored in them
        paths_reparam:  nodes are indexed according to their appearence
                        in the list of children
        """
        paths, paths_reparam = [], []
        self.get_paths_rec([], [], paths, paths_reparam)
        return paths, paths_reparam

    def trim(self, dict_of_leaves):
        """
        Remove all paths in which leaves are not in dict_of_leaves
        Returns:
            bool, whether or not path should be kept.
        """

        # If leaf node, return True (keep) if the value is on dict_of_leaves
        if len(self.children) == 0:
            return (self.data in dict_of_leaves)

        # Construct new list of children with children to keep.
        new_children = []
        for c in self.children[:]:
            if c.trim(dict_of_leaves):
                new_children.append(c)

        self.children = new_children

        if len(self.children) > 0: 
            return True

        return False

