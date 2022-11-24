import pathlib
import os
import random
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from utils.datasets.base import BaseDatasetAssemblyGraphs, AssemblyGraph


class AssemblyGraphs(BaseDatasetAssemblyGraphs):
    @staticmethod
    def num_classes(self):
        """Return the number of classes"""
        if self.args.node_dropping:
            return 6
        else:
            return 8

    def node_dim(self):
        """Return the node feature dimension"""
        return self.data[0]["assembly_graph"].x.shape[-1]

    def edge_dim(self):
        """Return the edge feature dimension"""
        return self.data[0]["assembly_graph"].e.shape[-1]

    def __init__(self, args, root_dir, split="train"):
        """
        Load the Assembly Graphs -> Perform Random Splitting + Saving the Splits -> Generate Data Samples
        """
        super().__init__(args)

        """Creating the train_val.txt and test.txt"""
        if split == "inference":
            path = pathlib.Path(root_dir)
            assemblies_ = os.listdir(path)
            assemblies = []

            # Check validity of the inference sample
            for i in tqdm(range(len(assemblies_))):
                if args.os == "Windows":
                    assemblies_[i] = f"{path}\\{assemblies_[i]}"
                    ag = AssemblyGraph(assemblies_[i] + '\\assembly.json', args)
                else:
                    assemblies_[i] = f"{path}/{assemblies_[i]}"
                    ag = AssemblyGraph(assemblies_[i] + '/assembly.json', args)

                ag = ag.get_graph_networkx()

                if args.node_dropping:
                    remove_nodes = []
                    for node in ag.nodes(data=True):
                        node_id = node[0]
                        node_data = node[-1]

                        if node_data["material"] == "Paint" or \
                                node_data["material"] == "Metal_Ferrous_Steel":
                            remove_nodes.append(node_id)

                    for node in remove_nodes:
                        ag.remove_node(node)

                """Drop graphs that are too small"""
                if ag.number_of_nodes() < 3 or ag.number_of_edges() < 2:
                    print(f"[Warning] Graph too small and are dropped: {assemblies_[i]}")
                else:
                    assemblies.append(assemblies_[i])

            if len(assemblies) == 0:
                print("[Error] All inference samples are too small and dropped!")
                exit(1)

            print(f"Loading {split} assemblies...")
            self.load_assemblies(assemblies, assemblies, vocab=args.vocab, inference=True)
            print("Done loading {} assemblies".format(len(self.data)))

        else:
            path = pathlib.Path(root_dir)
            assemblies_ = os.listdir(path)
            assemblies, assembly_ids = [], []
            all_cnt, too_small_cnt = 0, 0

            for i in tqdm(range(len(assemblies_)), desc="Preprocessing JSON files (dropping trivial assemblies)"):
                if len(assemblies_[i].split(".txt")) != 1:
                    # skip ".txt" files that are used for fix split
                    continue
                if args.os == "Windows":
                    assemblies_[i] = f"{path}\\{assemblies_[i]}"
                    ag = AssemblyGraph(assemblies_[i] + '\\assembly.json', args)
                else:
                    assemblies_[i] = f"{path}/{assemblies_[i]}"
                    ag = AssemblyGraph(assemblies_[i] + '/assembly.json', args)
                ag = ag.get_graph_networkx()
                all_cnt += 1

                """
                    Perform node-dropping of Default/Paint materials, to pre-eliminate graphs that are too small
                    (This is FIRST time of node-dropping: for eliminating graphs that are too small, for creating splits)
                """

                if args.node_dropping:
                    remove_nodes = []
                    for node in ag.nodes(data=True):
                        node_id = node[0]
                        node_data = node[-1]

                        if node_data["material"] == "Paint" or \
                                node_data["material"] == "Metal_Ferrous_Steel":
                            remove_nodes.append(node_id)

                    for node in remove_nodes:
                        ag.remove_node(node)

                """Drop graphs that are too small"""
                if ag.number_of_nodes() < 3 or ag.number_of_edges() < 2:
                    too_small_cnt += 1
                    continue

                # Now [assemblies] contains only those assemblies PATHS that are not too small, even after node dropping
                # Now [assembly_ids] contains only those assemblies IDS that are not too small, even after node dropping
                assemblies.append(assemblies_[i])
                if args.os == "Windows":
                    assembly_ids.append(assemblies_[i].split('\\')[-1])
                else:
                    assembly_ids.append(assemblies_[i].split('/')[-1])

            print(f"[Warning] Number of input graphs that are too small and are dropped: {too_small_cnt} / {all_cnt}")

            """Perform train test split"""

            if args.fixed_split:
                """Use pre-defined train test split (finding those that are not too small)"""
                print("[Train Test Split] FIXED split using predefined split (TXT files)")
                if args.os == "Windows":
                    with open(root_dir + '\\' + "assemblies_train_val.txt") as f:
                        lines = f.readlines()
                        train_val_assemblies = [root_dir + '\\' + line.strip() for line in lines
                                                if line.strip() in assembly_ids]

                    with open(args.dataset_path + '\\' + "assemblies_test.txt") as f:
                        lines = f.readlines()
                        test_assemblies = [root_dir + '\\' + line.strip() for line in lines
                                           if line.strip() in assembly_ids]
                else:
                    with open(root_dir + '/' + "assemblies_train_val.txt") as f:
                        lines = f.readlines()
                        train_val_assemblies = [root_dir + '/' + line.strip() for line in lines
                                                if line.strip() in assembly_ids]

                    with open(args.dataset_path + '/' + "assemblies_test.txt") as f:
                        lines = f.readlines()
                        test_assemblies = [root_dir + '/' + line.strip() for line in lines
                                           if line.strip() in assembly_ids]
            else:
                print("[Train Test Split] RANDOM split with global seed")
                random.shuffle(assemblies)
                train_percentage = 0.8
                train_val_assemblies = assemblies[:int(len(assemblies) * train_percentage)]
                test_assemblies = assemblies[int(len(assemblies) * train_percentage):]

            """Getting files for train, val, and test"""
            if split in ("train", "val"):
                train_assemblies, val_assemblies = train_test_split(train_val_assemblies, test_size=0.2,
                                                                    random_state=42)
                if split == "train":
                    file_paths = train_assemblies
                    args.train_set = file_paths
                else:
                    file_paths = val_assemblies
                    args.val_set = file_paths
            else:
                file_paths = test_assemblies
                args.test_set = file_paths

            print(f"Loading {split} assemblies...")
            self.load_assemblies(file_paths, assemblies)
            print("Done loading {} assemblies".format(len(self.data)))

    def _collate(self, batch):
        """Not doing anything"""
        collated = super()._collate(batch)
        # collated["label"] = torch.cat([x["label"] for x in batch], dim=0)
        return collated
