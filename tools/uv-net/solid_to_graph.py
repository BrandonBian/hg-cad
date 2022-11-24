import argparse
import pathlib

import dgl
import os
import numpy as np
import torch
from occwl.graph import face_adjacency
from occwl.io import load_step
from occwl.uvgrid import ugrid, uvgrid
from tqdm import tqdm
from multiprocessing.pool import Pool
from itertools import repeat
import signal
import shutil
import time


def build_graph(solid, curv_num_u_samples, surf_num_u_samples, surf_num_v_samples):
    # Build face adjacency graph with B-rep entities as node and edge features
    graph = face_adjacency(solid)

    # Compute the UV-grids for faces
    graph_face_feat = []
    for face_idx in graph.nodes:
        # Get the B-rep face
        face = graph.nodes[face_idx]["face"]
        # Compute UV-grids
        points = uvgrid(
            face, method="point", num_u=surf_num_u_samples, num_v=surf_num_v_samples
        )
        normals = uvgrid(
            face, method="normal", num_u=surf_num_u_samples, num_v=surf_num_v_samples
        )
        visibility_status = uvgrid(
            face, method="visibility_status", num_u=surf_num_u_samples, num_v=surf_num_v_samples
        )
        mask = np.logical_or(visibility_status == 0, visibility_status == 2)  # 0: Inside, 1: Outside, 2: On boundary
        # Concatenate channel-wise to form face feature tensor
        face_feat = np.concatenate((points, normals, mask), axis=-1)
        graph_face_feat.append(face_feat)

    graph_face_feat = np.asarray(graph_face_feat)

    # Compute the U-grids for edges
    graph_edge_feat = []
    for edge_idx in graph.edges:
        # Get the B-rep edge
        edge = graph.edges[edge_idx]["edge"]
        # Ignore dgenerate edges, e.g. at apex of cone
        if not edge.has_curve():
            continue
        # Compute U-grids
        points = ugrid(edge, method="point", num_u=curv_num_u_samples)
        tangents = ugrid(edge, method="tangent", num_u=curv_num_u_samples)
        # Concatenate channel-wise to form edge feature tensor
        edge_feat = np.concatenate((points, tangents), axis=-1)
        graph_edge_feat.append(edge_feat)
    graph_edge_feat = np.asarray(graph_edge_feat)

    # Convert face-adj graph to DGL format
    edges = list(graph.edges)
    src = [e[0] for e in edges]
    dst = [e[1] for e in edges]
    dgl_graph = dgl.graph((src, dst), num_nodes=len(graph.nodes))
    dgl_graph.ndata["x"] = torch.from_numpy(graph_face_feat)
    dgl_graph.edata["x"] = torch.from_numpy(graph_edge_feat)
    return dgl_graph


def process_one_file(arguments):
    fn, args = arguments
    fn_stem = fn.stem
    output_path = pathlib.Path(args.output)

    existing_files = os.listdir(output_path)
    if (fn_stem + ".bin") in existing_files:
        # skipping graphs that are already generated
        return

    try:
        solid = load_step(fn)[0]  # Assume there's one solid per file

        graph = build_graph(
            solid, args.curv_u_samples, args.surf_u_samples, args.surf_v_samples
        )
    except:
        print(f"WARNING: error processing file '{fn}', skipped!")
        return

    dgl.data.utils.save_graphs(str(output_path.joinpath(fn_stem + ".bin")), [graph])


def initializer():
    """Ignore CTRL+C in the worker process."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def preprocess(args):
    input_path = pathlib.Path(args.dataset)
    assemblies = os.listdir(input_path)

    if os.path.exists("./helpers/temp_in"):
        shutil.rmtree("./helpers/temp_in")
    os.mkdir("./helpers/temp_in")

    for assembly in tqdm(assemblies, desc="Preprocessing"):
        from_path = pathlib.Path(str(input_path) + '/' + assembly)
        steps = list(from_path.glob("*.step"))
        for step in steps:
            body_id = str(step).split('\\')[-1].split(".step")[0]
            if body_id == "assembly":
                os.remove(step)  # we shouldn't need the assembly.step file
                continue
            rename = f"{assembly}_sep_{body_id}.step"

            source = step
            dest = "./helpers/temp_in" + '/' + rename

            try:
                shutil.move(source, dest)
            except:
                print("wait")
                time.sleep(1)


def process(args):
    input_path = pathlib.Path("./helpers/temp_in")
    output_path = pathlib.Path("./helpers/temp_out")

    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.mkdir(output_path)

    step_files = list(input_path.glob("*.st*p"))
    pool = Pool(processes=args.num_processes, initializer=initializer)
    try:
        results = list(tqdm(pool.imap(process_one_file, zip(step_files, repeat(args))), total=len(step_files)))
    except KeyboardInterrupt:
        pool.terminate()
        pool.join()
    print(f"Processed {len(results)} files.")


def postprocess(args):
    input_path = pathlib.Path("./helpers/temp_out")
    for body in os.listdir(input_path):
        assembly_id = body.split("_sep_")[0]
        body_id = body.split("_sep_")[1].split(".bin")[0]

        source = f"./helpers/temp_out/{body}"
        dest = f"{args.dataset}/{assembly_id}/{body_id}.bin"
        shutil.move(source, dest)

    shutil.rmtree("./helpers/temp_in")
    shutil.rmtree("./helpers/temp_out")


def main():
    parser = argparse.ArgumentParser(
        "Convert solid models to face-adjacency graphs with UV-grid features"
    )

    parser.add_argument("dataset", type=str)
    parser.add_argument("--input", type=str, default="./helpers/temp_out", required=False)
    parser.add_argument("--output", type=str, default="./helpers/temp_out", required=False)

    parser.add_argument(
        "--curv_u_samples", type=int, default=10, help="Number of samples on each curve"
    )
    parser.add_argument(
        "--surf_u_samples",
        type=int,
        default=10,
        help="Number of samples on each surface along the u-direction",
    )
    parser.add_argument(
        "--surf_v_samples",
        type=int,
        default=10,
        help="Number of samples on each surface along the v-direction",
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=8,
        help="Number of processes to use",
    )
    args = parser.parse_args()

    preprocess(args)
    process(args)
    postprocess(args)


if __name__ == "__main__":
    main()
