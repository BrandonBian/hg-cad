from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
import torch_geometric
from torch import FloatTensor
import torch
import dgl
import numpy as np
import json
from dgl.data.utils import load_graphs
from utils.datasets import util
from tqdm import tqdm
from collections import Counter
from abc import abstractmethod
from pathlib import Path
from utils.datasets.process_assembly_graph import process_assembly_graph, get_vocab
from itertools import compress


def filtering(batched_body_graphs):
    """
        Providing mask and filtering for body_graphs (filter out those that are "NA")

        Return
            [filtered_body_graphs]
            [filtered_labels]
            [mask]
    """

    mask = [(element != "NA - bin file missing")
            and (element != "NA - no edge") and (element != "NA - too large") for element in batched_body_graphs]

    filtered_body_graphs = list(compress(batched_body_graphs, mask))

    return filtered_body_graphs, mask


class BaseDatasetAssemblyGraphs(Dataset):
    def __init__(self, args):
        self.data = []
        self.args = args
        self.weights = None

    @staticmethod
    @abstractmethod
    def num_classes(self):
        if self.args.node_dropping:
            return 6
        else:
            return 8

    def load_assemblies(self, file_paths, assemblies):
        """
            Input:
                [file_paths] list of paths of assemblies to be loaded (e.g., train / val / test assemblies)
                [assemblies] list of paths of ALL ASSEMBLIES loaded (i.e., train + val + test assemblies)
        """
        assembly_graphs = []
        dropped_cnt = 0

        for file_path in assemblies:
            if self.args.os == "Windows":
                ag = AssemblyGraph(f"{file_path}\\assembly.json", self.args)
            else:
                ag = AssemblyGraph(f"{file_path}/assembly.json", self.args)
            assembly_graph = ag.get_graph_networkx()

            """
                Perform node-dropping of Default/Paint materials, if applicable
                (This is SECOND time of node-dropping: for creation of vocab on entire dataset after dropping nodes)
            """
            if self.args.node_dropping:
                remove_nodes = []
                for node in assembly_graph.nodes(data=True):
                    node_id = node[0]
                    node_data = node[-1]

                    if node_data["material"] == "Paint" or \
                            node_data["material"] == "Metal_Ferrous_Steel":
                        remove_nodes.append(node_id)

                for node in remove_nodes:
                    assembly_graph.remove_node(node)

            assembly_graphs.append(assembly_graph)

        vocab, self.weights = get_vocab(
            assembly_graphs)  # Get vocab on ENTIRE dataset, to ensure correct one-hot encoding

        """Load Assembly Graphs"""
        for assembly in tqdm(file_paths):
            assembly, body_graphs, materials = self.load_one_graph(assembly, vocab)

            # Check if all body graphs in this assembly are invalid - if so, discard this assembly
            all_body_graphs_invalid = True
            for body_graph in body_graphs:
                if type(body_graph) != str:
                    all_body_graphs_invalid = False
            if all_body_graphs_invalid:
                dropped_cnt += 1
                continue

            self.data.append({"assembly_graph": assembly, "body_graphs": body_graphs, "labels": materials})

        """Obtain Some Statistics"""

        if self.args.node_dropping:
            mapping = {0: 'Metal_Aluminum', 1: 'Metal_Ferrous', 2: 'Metal_Non-Ferrous',
                       3: 'Other', 4: 'Plastic', 5: 'Wood'}
        else:
            mapping = {0: 'Metal_Aluminum', 1: 'Metal_Ferrous', 2: 'Metal_Ferrous_Steel', 3: 'Metal_Non-Ferrous',
                       4: 'Other', 5: 'Paint', 6: 'Plastic', 7: 'Wood'}
        ground_truth = [int(label) for sample in self.data for label in sample["labels"]]
        ground_truth_counter = Counter(ground_truth)
        ground_truth_dict = {}

        for label, freq in ground_truth_counter.most_common():
            ground_truth_dict[mapping[label]] = freq

        print("Ground truth distribution:", ground_truth_dict)
        print("Number of assemblies with no valid body graphs, thus dropped:", dropped_cnt)

    def load_one_graph(self, file_path, vocab):
        """
            Input:
                [file_path]: a single path to a single assembly
                [vocab]: the vocab dictionary generated from the entire dataset, for one-hot encoding features

            Return:
        """

        if self.args.os == "Windows":
            ag = AssemblyGraph(f"{file_path}\\assembly.json", self.args)  # obtain assembly graph from JSON
        else:
            ag = AssemblyGraph(f"{file_path}/assembly.json", self.args)  # obtain assembly graph from JSON
        assembly_graph = ag.get_graph_networkx()  # obtain NetworkX DiGraph representation

        """
            Node dropping, if applicable
            (This is THIRD time of node-dropping: for dropping the node per graph before encoding its features)
        """
        if self.args.node_dropping:
            remove_nodes = []
            for node in assembly_graph.nodes(data=True):
                node_id = node[0]
                node_data = node[-1]

                if node_data["material"] == "Paint" or \
                        node_data["material"] == "Metal_Ferrous_Steel":
                    remove_nodes.append(node_id)

            for node in remove_nodes:
                assembly_graph.remove_node(node)

        assembly_graph, body_graphs, materials = process_assembly_graph(assembly_graph, vocab, file_path, self.args)

        return assembly_graph, body_graphs, materials

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        # if self.random_rotate:
        #     rotation = util.get_random_rotation()
        #     sample["graph"].ndata["x"] = util.rotate_uvgrid(sample["graph"].ndata["x"], rotation)
        #     sample["graph"].edata["x"] = util.rotate_uvgrid(sample["graph"].edata["x"], rotation)
        return sample

    def _collate(self, batch):
        batched_assembly_graph = torch_geometric.data.Batch.from_data_list(
            [sample["assembly_graph"] for sample in batch])
        batched_body_graphs = [body_graph for sample in batch for body_graph in sample["body_graphs"]]
        batched_labels = torch.tensor([label for sample in batch for label in sample["labels"]])

        # Creating batch for body_graphs, with masking of NA nodes
        batched_body_graphs, mask = filtering(batched_body_graphs)
        batched_body_graphs = dgl.batch(batched_body_graphs)

        return {"assembly_graph": batched_assembly_graph, "body_graphs": batched_body_graphs,
                "labels": batched_labels, "mask": mask, "weights": self.weights}

    def get_dataloader(self, batch_size=128, shuffle=True, num_workers=6):
        """
            torch.utils.data.DataLoader -> can overload _collate(), but requires all tensors
            torch_geometric.loader.DataLoader -> can't overload _collate(), can add "kwargs", but wrong behavior for non-tensor
        """

        return torch.utils.data.DataLoader(
            self.data,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self._collate,
            drop_last=True,
            num_workers=num_workers)


class BaseDataset(Dataset):
    @staticmethod
    @abstractmethod
    def num_classes():
        pass

    def load_graphs(self, file_paths, center_and_scale=True):
        no_edge, too_many_nodes = 0, 0
        self.data = []
        for fn in tqdm(file_paths):
            if not fn.exists():
                continue
            sample = self.load_one_graph(fn)
            if sample is None:
                continue
            if sample["graph"].edata["x"].size(0) == 0:
                # Catch the case of graphs with no edges
                no_edge += 1
                continue
            if dgl.DGLGraph.number_of_nodes(sample["graph"]) > 150:
                # Skip graphs that have too many nodes
                too_many_nodes += 1
                continue
            self.data.append(sample)
        if center_and_scale:
            self.center_and_scale()
        self.convert_to_float32()
        print(f"Number of graphs skipped (no edge + too many nodes): {no_edge} + {too_many_nodes} / {len(file_paths)}")

    def load_one_graph(self, file_path):
        graph = load_graphs(str(file_path))[0][0]
        sample = {"graph": graph, "filename": file_path.stem}
        return sample

    def center_and_scale(self):
        for i in range(len(self.data)):
            self.data[i]["graph"].ndata["x"], center, scale = util.center_and_scale_uvgrid(
                self.data[i]["graph"].ndata["x"], return_center_scale=True
            )
            self.data[i]["graph"].edata["x"][..., :3] -= center
            self.data[i]["graph"].edata["x"][..., :3] *= scale

    def convert_to_float32(self):
        for i in range(len(self.data)):
            self.data[i]["graph"].ndata["x"] = self.data[i]["graph"].ndata["x"].type(FloatTensor)
            self.data[i]["graph"].edata["x"] = self.data[i]["graph"].edata["x"].type(FloatTensor)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        # if self.random_rotate:
        #     rotation = util.get_random_rotation()
        #     sample["graph"].ndata["x"] = util.rotate_uvgrid(sample["graph"].ndata["x"], rotation)
        #     sample["graph"].edata["x"] = util.rotate_uvgrid(sample["graph"].edata["x"], rotation)
        return sample

    def _collate(self, batch):
        batched_graph = dgl.batch([sample["graph"] for sample in batch])
        batched_filenames = [sample["filename"] for sample in batch]
        return {"graph": batched_graph, "filename": batched_filenames}

    def get_dataloader(self, batch_size=128, shuffle=True, num_workers=6):
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self._collate,
            num_workers=num_workers,  # Can be set to non-zero on Linux
            drop_last=True
        )


class AssemblyGraph:
    """
        Construct a graph representing an assembly with connectivity
        between B-Rep bodies with joints and contact surfaces

        Relevant NODE features:
            "body_uuid", "occ_id"
            "material"
            "body_name_embedding", "occ_name_embedding"
            "body_area", "body_volume"
            "center_x", "center_y", "center_z"
            (MVCNN embeddings: "visual_embedding")
            (ResNet 2D image fingerprint: "image_fingerprint")
            "occurrence_area", "occurrence_volume"
            "global_features"

        Relevant EDGE features:
            "id"
            "type"
    """

    def __init__(self, assembly_data, args):

        self.args = args
        if self.args.os == "Windows":
            self.assembly_id = str(assembly_data).split('\\')[-2]
        else:
            self.assembly_id = str(assembly_data).split('/')[-2]

        if isinstance(assembly_data, dict):
            self.assembly_data = assembly_data
        else:
            if isinstance(assembly_data, str):
                assembly_file = Path(assembly_data)
            else:
                assembly_file = assembly_data
            assert assembly_file.exists()
            with open(assembly_file, "r", encoding="utf-8") as f:
                self.assembly_data = json.load(f)

        # self.args = args
        self.graph_nodes = []
        self.graph_links = []
        self.graph_node_ids = set()
        self.depth = 0

    def get_graph_data(self):
        """Get the graph data as a list of nodes and links"""

        self.graph_nodes = []
        self.graph_links = []
        self.graph_node_ids = set()

        self.populate_graph_nodes()
        self.populate_graph_links()
        self.populate_graph_shared_occ_links()
        self.populate_graph_global_features()

        return self.graph_nodes, self.graph_links, self.depth, None

    def get_graph_networkx(self):
        """Get a networkx graph"""
        graph_data = {
            "directed": True,
            "multigraph": False,
            "graph": {},
            "nodes": [],
            "links": [],
        }
        graph_data["nodes"], graph_data["links"], _, _ = self.get_graph_data()
        from networkx.readwrite import json_graph
        return json_graph.node_link_graph(graph_data)

    def get_node_label_dict(self, attribute="occurrence_name"):
        """Get a dictionary mapping from node ids to a given attribute"""
        label_dict = {}
        if len(self.graph_nodes) == 0:
            return label_dict
        for node in self.graph_nodes:
            node_id = node["id"]
            if attribute in node:
                node_att = node[attribute]
            else:
                node_att = node["body_name"]
            label_dict[node_id] = node_att
        return label_dict

    def get_graph_links(self):
        """Get the links"""

        graph_data = {
            "directed": True,
            "multigraph": False,
            "graph": {},
            "nodes": [],
            "links": [],
        }
        graph_data["nodes"], graph_data["links"], _, _ = self.get_graph_data()
        return json.dumps(graph_data["links"])

    def populate_graph_nodes(self):
        """
        Recursively traverse the assembly tree
        and generate a flat set of graph nodes from bodies
        """
        root_component_uuid = self.assembly_data["root"]["component"]
        root_component = self.assembly_data["components"][root_component_uuid]

        if "bodies" in root_component:
            for body_uuid in root_component["bodies"]:
                node_data = self.get_graph_node_data(body_uuid)
                self.graph_nodes.append(node_data)

        tree_root = self.assembly_data["tree"]["root"]
        root_transform = np.identity(4)
        self.walk_tree(tree_root, root_transform)

        total_nodes = len(self.graph_nodes)
        self.graph_node_ids = set([f["id"] for f in self.graph_nodes])
        assert total_nodes == len(self.graph_node_ids), "Duplicate node ids found"

    def populate_graph_links(self):
        """Create links in the graph between bodies with joints and contacts"""
        if "joints" in self.assembly_data:
            self.populate_graph_joint_links()
        if "as_built_joints" in self.assembly_data:
            self.populate_graph_as_built_joint_links()
        if "contacts" in self.assembly_data:
            self.populate_graph_contact_links()

    def populate_graph_shared_occ_links(self):
        for source in self.graph_nodes:
            for target in self.graph_nodes:
                if source == target:
                    continue
                if source["occ_id"] == target["occ_id"]:
                    source_id = source["id"]
                    target_id = target["id"]
                    link_data = {"id": f"{source_id}>{target_id}", "source": source_id, "target": target_id,
                                 "type": "Shared Occurrence"}
                    self.graph_links.append(link_data)

    def populate_graph_global_features(self):
        for node in self.graph_nodes:
            node["global_features"] = {}
            node["global_features"]["edge_count"] = self.assembly_data["properties"]["edge_count"]
            node["global_features"]["face_count"] = self.assembly_data["properties"]["face_count"]
            node["global_features"]["loop_count"] = self.assembly_data["properties"]["loop_count"]
            node["global_features"]["shell_count"] = self.assembly_data["properties"]["shell_count"]
            node["global_features"]["vertex_count"] = self.assembly_data["properties"]["vertex_count"]
            node["global_features"]["volume"] = self.assembly_data["properties"]["volume"]
            node["global_features"]["center_x"] = self.assembly_data["properties"]["center_of_mass"]["x"]
            node["global_features"]["center_y"] = self.assembly_data["properties"]["center_of_mass"]["y"]
            node["global_features"]["center_z"] = self.assembly_data["properties"]["center_of_mass"]["z"]
            # compatibility with older datasets: older datasets may not have the following global features
            node["global_features"]["products"] = self.assembly_data["properties"]["products"]
            node["global_features"]["categories"] = self.assembly_data["properties"]["categories"]
            node["global_features"]["industries"] = self.assembly_data["properties"]["industries"]
            node["global_features"]["likes_count"] = self.assembly_data["properties"]["likes_count"]
            node["global_features"]["comments_count"] = self.assembly_data["properties"]["comments_count"]
            node["global_features"]["views_count"] = self.assembly_data["properties"]["views_count"]

    def walk_tree(self, occ_tree, occ_transform):
        """Recursively walk the occurrence tree"""
        self.depth = 1

        for occ_uuid, occ_sub_tree in occ_tree.items():
            self.depth += 1
            occ = self.assembly_data["occurrences"][occ_uuid]
            if not occ["is_visible"]:
                continue

            occ_sub_transform = occ_transform @ self.transform_to_matrix(occ["transform"])

            if "bodies" in occ:
                for occ_body_uuid, occ_body in occ["bodies"].items():
                    if not occ_body["is_visible"]:
                        continue
                    node_data = self.get_graph_node_data(
                        occ_body_uuid,
                        occ_uuid,
                        occ,
                        occ_sub_transform
                    )
                    self.graph_nodes.append(node_data)
            self.walk_tree(occ_sub_tree, occ_sub_transform)

    def get_graph_node_data(self, body_uuid, occ_uuid=None, occ=None, transform=None):
        """Add a body as a graph node"""

        """We only want [numerical features] + [material ground truths (str)] + [UUID (str)]"""

        # General Information
        body = self.assembly_data["bodies"][body_uuid]
        node_data = {}
        if occ_uuid is None:
            body_id = body_uuid
        else:
            body_id = f"{occ_uuid}_{body_uuid}"

        node_data["body_uuid"] = body_uuid
        node_data["occ_id"] = occ_uuid
        node_data["id"] = body_id

        # Material (Ground Truths) Information
        node_data["material"] = body["material_category_simple"]

        # Body Information
        node_data["body_name_embedding"] = body["body_name_embedding"]
        node_data["body_area"] = body["physical_properties"]["area"]
        node_data["body_volume"] = body["physical_properties"]["volume"]
        node_data["center_x"] = body["physical_properties"]["center_of_mass"]["x"]
        node_data["center_y"] = body["physical_properties"]["center_of_mass"]["y"]
        node_data["center_z"] = body["physical_properties"]["center_of_mass"]["z"]

        if self.args.MVCNN_embedding:
            if body["visual_embedding"] is None:
                node_data["MVCNN_embedding"] = [0] * 512
            else:
                node_data["MVCNN_embedding"] = body["visual_embedding"]

        if self.args.image_fingerprint:
            node_data["image_fingerprint"] = body["similarity_stats"]

        # Occurrence Information
        if occ:
            node_data["occurrence_area"] = occ["physical_properties"]["area"]
            node_data["occurrence_volume"] = occ["physical_properties"]["volume"]
            node_data["occ_name_embedding"] = occ["occ_name_embedding"]

        else:
            node_data["occurrence_area"] = 0
            node_data["occurrence_volume"] = 0
            node_data["occ_name_embedding"] = [0] * 600

        return node_data

    def populate_graph_joint_links(self):
        """Populate directed links between bodies with joints"""
        if self.assembly_data["joints"] is None:
            pass
        else:
            for joint_uuid, joint in self.assembly_data["joints"].items():
                try:

                    ent1 = joint["geometry_or_origin_one"]["entity_one"]
                    ent2 = joint["geometry_or_origin_two"]["entity_one"]

                    body1_visible = self.is_body_visible(ent1)
                    body2_visible = self.is_body_visible(ent2)
                    if not body1_visible or not body2_visible:
                        continue
                    link_data = self.get_graph_link_data(ent1, ent2)
                    link_data["type"] = "Joint"
                    link_data["joint_type"] = joint["joint_motion"]["joint_type"]
                    self.graph_links.append(link_data)
                except:
                    continue

    def populate_graph_as_built_joint_links(self):
        """Populate directed links between bodies with as built joints"""
        if self.assembly_data["as_built_joints"] is None:
            pass
        else:
            for joint_uuid, joint in self.assembly_data["as_built_joints"].items():
                geo_ent = None
                geo_ent_id = None

                if "joint_geometry" in joint:
                    if "entity_one" in joint["joint_geometry"]:
                        geo_ent = joint["joint_geometry"]["entity_one"]
                        geo_ent_id = self.get_link_entity_id(geo_ent)

                occ1 = joint["occurrence_one"]
                occ2 = joint["occurrence_two"]
                body1 = None
                body2 = None
                if geo_ent is not None and "occurrence" in geo_ent:
                    if geo_ent["occurrence"] == occ1:
                        body1 = geo_ent["body"]
                    elif geo_ent["occurrence"] == occ2:
                        body2 = geo_ent["body"]

                if body1 is None:
                    body1 = self.get_occurrence_body_uuid(occ1)
                    if body1 is None:
                        continue
                if body2 is None:
                    body2 = self.get_occurrence_body_uuid(occ2)
                    if body2 is None:
                        continue

                body1_visible = self.is_body_visible(body_uuid=body1, occurrence_uuid=occ1)
                body2_visible = self.is_body_visible(body_uuid=body2, occurrence_uuid=occ2)
                if not body1_visible or not body2_visible:
                    continue
                ent1 = f"{occ1}_{body1}"
                ent2 = f"{occ2}_{body2}"
                link_id = f"{ent1}>{ent2}"
                link_data = {}
                link_data["id"] = link_id
                link_data["source"] = ent1
                assert link_data["source"] in self.graph_node_ids, "Link source id doesn't exist in nodes"
                link_data["target"] = ent2
                assert link_data["target"] in self.graph_node_ids, "Link target id doesn't exist in nodes"
                link_data["type"] = "AsBuiltJoint"
                link_data["joint_type"] = joint["joint_motion"]["joint_type"]
                self.graph_links.append(link_data)

    def populate_graph_contact_links(self):
        """Populate undirected links between bodies in contact"""
        if self.assembly_data["contacts"] == None:
            pass
        else:
            for contact in self.assembly_data["contacts"]:
                ent1 = contact["entity_one"]
                ent2 = contact["entity_two"]

                body1_visible = self.is_body_visible(ent1)
                body2_visible = self.is_body_visible(ent2)
                if not body1_visible or not body2_visible:
                    continue
                link_data = self.get_graph_link_data(ent1, ent2)
                link_data["type"] = "Contact"
                self.graph_links.append(link_data)

                link_data = self.get_graph_link_data(ent2, ent1)
                link_data["type"] = "Contact"
                self.graph_links.append(link_data)

    def get_graph_link_data(self, entity_one, entity_two):
        """Get the common data for a graph link from a joint or contact"""
        link_data = {"id": self.get_link_id(entity_one, entity_two), "source": self.get_link_entity_id(entity_one)}
        assert link_data["source"] in self.graph_node_ids, "Link source id doesn't exist in nodes"
        link_data["target"] = self.get_link_entity_id(entity_two)
        assert link_data["target"] in self.graph_node_ids, "Link target id doesn't exist in nodes"
        return link_data

    def get_link_id(self, entity_one, entity_two):
        """Get a unique id for a link"""
        ent1_id = self.get_link_entity_id(entity_one)
        ent2_id = self.get_link_entity_id(entity_two)
        return f"{ent1_id}>{ent2_id}"

    def get_link_entity_id(self, entity):
        """Get a unique id for one side of a link"""
        if "occurrence" in entity:
            return f"{entity['occurrence']}_{entity['body']}"
        else:
            return entity["body"]

    def get_occurrence_body_uuid(self, occurrence_uuid):
        """Get the body uuid from an occurrence"""
        occ = self.assembly_data["occurrences"][occurrence_uuid]

        if "bodies" not in occ:
            return None
        if len(occ["bodies"]) != 1:
            return None

        return next(iter(occ["bodies"]))

    def is_body_visible(self, entity=None, body_uuid=None, occurrence_uuid=None):
        """Check if a body is visible"""
        if body_uuid is None:
            body_uuid = entity["body"]
        if occurrence_uuid is None:

            if "occurrence" not in entity:
                body = self.assembly_data["root"]["bodies"][body_uuid]
                return body["is_visible"]

            occurrence_uuid = entity["occurrence"]
        occ = self.assembly_data["occurrences"][occurrence_uuid]
        if not occ["is_visible"]:
            return False
        body = occ["bodies"][body_uuid]
        return body["is_visible"]

    def transform_to_matrix(self, transform=None):
        """
        Convert a transform dict into a
        4x4 affine transformation matrix
        """
        if transform is None:
            return np.identity(4)
        x_axis = self.transform_vector_to_np(transform["x_axis"])
        y_axis = self.transform_vector_to_np(transform["y_axis"])
        z_axis = self.transform_vector_to_np(transform["z_axis"])
        translation = self.transform_vector_to_np(transform["origin"])
        translation[3] = 1.0
        return np.transpose(np.stack([x_axis, y_axis, z_axis, translation]))

    def transform_vector_to_np(self, vector):
        x = vector["x"]
        y = vector["y"]
        z = vector["z"]
        h = 0.0
        return np.array([x, y, z, h])
