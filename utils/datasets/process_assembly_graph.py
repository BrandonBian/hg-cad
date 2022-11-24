import os
from collections import Counter
from torch_geometric.data import Data
import torch
import torch.nn as nn
import networkx as nx
import dgl
from dgl.data.utils import load_graphs
from sklearn import preprocessing
from utils.datasets import util


def standard_scalar_transformation(tensor):
    """Perform standard scalar transformation"""
    scaler = preprocessing.StandardScaler().fit(tensor)
    scaled = scaler.transform(tensor)
    scaled = torch.from_numpy(scaled)
    return scaled


def get_vocab(assembly_graphs):
    """
        Obtain the vocab dictionary for helping the one-hot encoding in "process_assembly_graph()"

        Input:
            A list of NetworkX DiGraph assembly graphs
        Output:
            vocab - a dictionary (for supporting one-hot encoding in "process_assembly_graph()")
            weights - a torch tensor of weights of materials labels (for weighted Cross Entropy)
    """

    # Vocab - only for categorical (string) features
    vocab = {
        # Node features
        'material': set(),

        # Edge features
        'edge_type': set(),

        # Global features
        'products': set(),
        'categories': set(),
        'industries': set(),
    }

    material_count = []

    for graph in assembly_graphs:
        for node in graph.nodes(data=True):
            node = node[-1]

            vocab['material'].add(node['material'])
            vocab['products'].update(node["global_features"]["products"])
            vocab['categories'].update(node["global_features"]["categories"])
            vocab['industries'].update(node["global_features"]["industries"])

            material_count.append(node['material'])

        for edge in graph.edges(data=True):
            edge = edge[-1]
            if 'type' in edge:
                vocab['edge_type'].add(edge['type'])

    for k, v in vocab.items():
        vocab[k] = {s: idx for idx, s in enumerate(sorted(v))}

    material_count = [vocab['material'][l] for l in material_count]
    material_w = [0.] * len(vocab['material'])

    for k, v in Counter(material_count).items():
        material_w[k] = len(material_count) / v

    weights = torch.tensor(material_w)

    return vocab, weights


def process_body_graphs(graph, file_path, args):
    """
        Obtain a list of DGL graphs for the bodies of the assembly

        Input:
            [file_path] directory to the assembly

        Output
            [body_graphs] a list of DGL format body graphs (should be same order as for process_assembly_graph())
    """
    body_bins, body_graphs = [], []

    for file in os.listdir(file_path):
        # get all BIN files in assembly directory
        if ".bin" in file:
            if args.os == "Windows":
                body_bins.append(f"{file_path}\\{file}")
            else:
                body_bins.append(f"{file_path}/{file}")

    for node in graph.nodes(data=True):
        """Note: the for loop should preserve node traversing order"""

        node_id = node[0].split("_")[-1]  # node_id = "[occ_id]_[body_id]", if [occ_id] applicable
        if args.os == "Windows":
            all_node_ids = [x.split('\\')[-1].split(".bin")[0] for x in body_bins]
        else:
            all_node_ids = [x.split('/')[-1].split(".bin")[0] for x in body_bins]

        if node_id not in all_node_ids:
            # if we cannot find a corresponding BIN file for this body
            # possible reason: failed to be converted from the STEP file (in solid_to_graph.py)
            body_graphs.append("NA - bin file missing")
        else:
            if args.os == "Windows":
                body_bin = f"{file_path}\\{node_id}.bin"
            else:
                body_bin = f"{file_path}/{node_id}.bin"
            body_graph = load_graphs(body_bin)[0][0]

            # Additional processing of each body graphs in assembly

            if body_graph.edata["x"].size(0) == 0:
                # No edge
                body_graphs.append("NA - no edge")

            elif dgl.DGLGraph.number_of_nodes(body_graph) > 50:  # TODO: increase if applicable
                # Too many nodes
                body_graphs.append("NA - too large")

            else:
                # Good
                body_graphs.append(body_graph)

    """Center and Scale"""
    for i in range(len(body_graphs)):
        if body_graphs[i] in ("NA - no edge", "NA - too large", "NA - bin file missing"):
            continue
        body_graphs[i].ndata["x"], center, scale = util.center_and_scale_uvgrid(body_graphs[i].ndata["x"],
                                                                                return_center_scale=True)

        body_graphs[i].edata["x"][..., :3] -= center
        body_graphs[i].edata["x"][..., :3] *= scale

    """Convert to Float-32"""
    for i in range(len(body_graphs)):
        if body_graphs[i] in ("NA - no edge", "NA - too large", "NA - bin file missing"):
            continue
        body_graphs[i].ndata["x"] = body_graphs[i].ndata["x"].type(torch.FloatTensor)

        body_graphs[i].edata["x"] = body_graphs[i].edata["x"].type(torch.FloatTensor)

    assert len(body_graphs) == graph.number_of_nodes()

    return body_graphs


def process_assembly_graph(graph, vocab, file_path, args):
    """
        Process the assembly graph, encoding all features (e.g., One-Hot and Standard Scalar), and concatenate

        Input:
            [graph] NetworkX Digraph of assembly graph
            [vocab] Vocab dictionary for one-hot encoding of categorical (string) features
            [file_path] directory to the assembly

        Output:
            [data] A Torch Geometric Data() instance, with the following elements:
                1. nodes = Concatenated node features: [num of nodes] x [length of concatenated node features]
                2. edges = Concatenated edge features: [num of edges] x [length of concatenated edge features]
                3. edge_index = Adjacency list: 2 x [num of edges]
            [body_graphs] A list of DGL body graphs corresponding to bodies of this assembly (preserving order)
            [material] A list of Material ground truth labels (one-hot): [num of nodes] x 1
            [body_ids] A list of body ids: [num of nodes] x 1
    """

    """Obtain DGL Body Graphs"""
    body_graphs = process_body_graphs(graph, file_path, args)

    nodes, edges = [], []
    mappings = {n: idx for idx, n in enumerate(graph.nodes())}  # Change the node names from UUIDs to indices
    graph = nx.relabel_nodes(graph, mappings)

    """Global Features"""

    global_edge_count_tensor = torch.tensor(
        [graph.nodes(data=True)[1]['global_features']["edge_count"]], dtype=torch.float)
    global_face_count_tensor = torch.tensor(
        [graph.nodes(data=True)[1]['global_features']["face_count"]], dtype=torch.float)
    global_loop_count_tensor = torch.tensor(
        [graph.nodes(data=True)[1]['global_features']["loop_count"]], dtype=torch.float)
    global_shell_count_tensor = torch.tensor(
        [graph.nodes(data=True)[1]['global_features']["shell_count"]], dtype=torch.float)
    global_vertex_count_tensor = torch.tensor(
        [graph.nodes(data=True)[1]['global_features']["vertex_count"]], dtype=torch.float)
    global_volume_tensor = torch.tensor(
        [graph.nodes(data=True)[1]['global_features']["volume"]], dtype=torch.float)
    global_center_x_tensor = torch.tensor(
        [graph.nodes(data=True)[1]['global_features']["center_x"]], dtype=torch.float)
    global_center_y_tensor = torch.tensor(
        [graph.nodes(data=True)[1]['global_features']["center_y"]], dtype=torch.float)
    global_center_z_tensor = torch.tensor(
        [graph.nodes(data=True)[1]['global_features']["center_z"]], dtype=torch.float)
    products = graph.nodes(data=True)[1]["global_features"]["products"]
    global_products = nn.functional.one_hot(torch.tensor([vocab['products'][prod] for prod in products]),
                                            num_classes=len(vocab['products'])).sum(dim=0).float()
    categories = graph.nodes(data=True)[1]["global_features"]["categories"]
    global_categories = nn.functional.one_hot(torch.tensor([vocab['categories'][cat] for cat in categories]),
                                              num_classes=len(vocab['categories'])).sum(dim=0).float()
    industries = graph.nodes(data=True)[1]["global_features"]["industries"]
    if len(industries) > 0:
        global_industries = nn.functional.one_hot(
            torch.tensor([vocab['industries'][ind] for ind in industries]),
            num_classes=len(vocab['industries'])).sum(dim=0).float()
    else:
        global_industries = torch.tensor(len(vocab['industries']) * [0]).long()
    global_likes_count = torch.tensor([graph.nodes(data=True)[1]['global_features']["likes_count"]],
                                      dtype=torch.float)
    global_comments_count = torch.tensor([graph.nodes(data=True)[1]['global_features']["comments_count"]],
                                         dtype=torch.float)
    global_views_count = torch.tensor([graph.nodes(data=True)[1]['global_features']["views_count"]],
                                      dtype=torch.float)

    global_features_tensor = torch.cat((
        global_edge_count_tensor,
        global_face_count_tensor,
        global_loop_count_tensor,
        global_shell_count_tensor,
        global_vertex_count_tensor,
        global_volume_tensor,
        global_center_x_tensor,
        global_center_y_tensor,
        global_center_z_tensor,
        global_products,
        global_categories,
        global_industries,
        global_likes_count,
        global_comments_count,
        global_views_count
    ), -1).reshape(-1, 1)

    global_features_scaler = preprocessing.StandardScaler().fit(global_features_tensor)
    global_features_scaled = torch.from_numpy(global_features_scaler.transform(global_features_tensor))
    global_features_scaled = global_features_scaled.transpose(0, 1).repeat(len(graph.nodes), 1)

    """Note: the for loop should preserve node traversing order"""

    """Occurrence Area"""
    occurrence_area_tensor = torch.tensor([[n[-1]['occurrence_area']] for n in graph.nodes(data=True)],
                                          dtype=torch.float)
    occurrence_area_scaled = standard_scalar_transformation(occurrence_area_tensor)

    """Occurrence Volume"""
    occurrence_volume_tensor = torch.tensor([[n[-1]['occurrence_volume']] for n in graph.nodes(data=True)],
                                            dtype=torch.float)
    occurrence_volume_scaled = standard_scalar_transformation(occurrence_volume_tensor)

    """Technet Embeddings"""
    body_name_embeddings = torch.tensor([[n[-1]['body_name_embedding']] for n in graph.nodes(data=True)],
                                        dtype=torch.float)

    occ_name_embeddings = torch.tensor([[n[-1]['occ_name_embedding']] for n in graph.nodes(data=True)],
                                       dtype=torch.float)

    body_name_embeddings = torch.squeeze(body_name_embeddings)
    occ_name_embeddings = torch.squeeze(occ_name_embeddings)

    body_name_embeddings_scaled = standard_scalar_transformation(body_name_embeddings)
    occ_name_embeddings_scaled = standard_scalar_transformation(occ_name_embeddings)

    """Center of Mass"""
    center_x = torch.tensor([[n[-1]['center_x']] for n in graph.nodes(data=True)], dtype=torch.float)
    center_y = torch.tensor([[n[-1]['center_y']] for n in graph.nodes(data=True)], dtype=torch.float)
    center_z = torch.tensor([[n[-1]['center_z']] for n in graph.nodes(data=True)], dtype=torch.float)

    center_x_scaled = standard_scalar_transformation(center_x)
    center_y_scaled = standard_scalar_transformation(center_y)
    center_z_scaled = standard_scalar_transformation(center_z)

    """Body Area"""
    body_area_tensor = torch.tensor([[n[-1]['body_area']] for n in graph.nodes(data=True)], dtype=torch.float)
    body_area_scaled = standard_scalar_transformation(body_area_tensor)

    """Body Volume"""
    body_volume_tensor = torch.tensor([[n[-1]['body_volume']] for n in graph.nodes(data=True)], dtype=torch.float)
    body_volume_scaled = standard_scalar_transformation(body_volume_tensor)

    """2D Image Fingerprint"""
    if args.image_fingerprint:
        fingerprint_tensor = torch.tensor([[n[-1]['image_fingerprint']] for n in graph.nodes(data=True)],
                                          dtype=torch.float)
        fingerprint_tensor = torch.squeeze(fingerprint_tensor)
        fingerprint_scaled = standard_scalar_transformation(fingerprint_tensor)

    """MVCNN Visual Embedding"""
    if args.MVCNN_embedding:
        MVCNN_embedding = torch.tensor([[n[-1]['MVCNN_embedding']] for n in graph.nodes(data=True)],
                                       dtype=torch.float)
        MVCNN_embedding_tensor = torch.squeeze(MVCNN_embedding)
        MVCNN_embedding_scaled = standard_scalar_transformation(MVCNN_embedding_tensor)

    """Ablation (if applicable)"""
    features_to_include = []

    if "global_features" not in args.ablation:
        features_to_include.append(global_features_scaled)

    if "body_name" not in args.ablation:
        features_to_include.append(body_name_embeddings_scaled)

    if "occ_name" not in args.ablation:
        features_to_include.append(occ_name_embeddings_scaled)

    if "center_of_mass" not in args.ablation:
        features_to_include.append(center_x_scaled)
        features_to_include.append(center_y_scaled)
        features_to_include.append(center_z_scaled)

    if "body_physical_properties" not in args.ablation:
        features_to_include.append(body_area_scaled)
        features_to_include.append(body_volume_scaled)

    if "occ_physical_properties" not in args.ablation:
        features_to_include.append(occurrence_area_scaled)
        features_to_include.append(occurrence_volume_scaled)

    if args.image_fingerprint and "image_fingerprint" not in args.ablation:
        features_to_include.append(fingerprint_scaled)

    if args.MVCNN_embedding and "MVCNN_embedding" not in args.ablation:
        features_to_include.append(MVCNN_embedding_scaled)

    """Node Feature Concatenation"""
    num_features = len(features_to_include)
    nodes = features_to_include.pop(0)

    for i in range(num_features - 1):
        feature = features_to_include.pop(0)
        nodes = torch.cat((nodes, feature), dim=-1)

    assert (len(features_to_include) == 0)  # sanity check

    """Material Ground Truths"""
    material = torch.tensor([vocab[f'material'][n[-1][f'material']] for n in graph.nodes(data=True)])

    """Body IDs"""
    body_ids = [n[-1]["body_uuid"] for n in graph.nodes(data=True)]

    """Edge Feature"""
    for edge in graph.edges(data=True):
        edges.append(torch.zeros(2 * len(vocab['edge_type']) + 1))

        if len(edge[-1]) == 0:
            edges[-1][0] = 1.
        else:
            if 'type' in edge[-1]:
                edges[-1][vocab['edge_type'][edge[-1]['type']] + 1] = 1.

    edges = torch.stack(edges)  # edge_attr

    """Adjacency List"""
    edge_index = torch.tensor([[e[0], e[1]] for e in graph.edges()]).transpose(1, 0)  # edge_index

    # 1. nodes = Concatenated node features: [num of nodes] x [length of concatenated node features]
    # 2. edges = Concatenated edge features: [num of edges] x [length of concatenated edge features]
    # 3. edge_index = Adjacency list: 2 x [num of edges]

    data = Data(x=nodes, edge_index=edge_index, e=edges)

    # ~ body_graphs = List of DGL body graphs: [num of nodes] x 1
    # ~ material = Material ground truth labels (one-hot): [num of nodes] x 1
    # ~ body_ids = List of body ids: [num of nodes] x 1

    return data, body_graphs, material, body_ids
