import argparse
import pathlib
import time
import os
import random
import pickle

import torch
import gc
import numpy
import matplotlib.pyplot as plt
import platform
import multiprocessing

from tqdm import tqdm
from pathlib import Path
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from pytorch_lightning.utilities.seed import seed_everything

from utils.models.models import ClassificationGNN
from utils.datasets.assemblygraphs import AssemblyGraphs


def get_parser():
    """Obtain argument parser"""

    """Customized Parameters"""
    parser = argparse.ArgumentParser("UV-Net solid model classification")
    parser.add_argument("type", choices=("single_sample", "multiple_sample"), default="multiple",
                        help="single or multiple inference")
    parser.add_argument("--inference_sample", type=str, help="Path to assembly sample to perform inference")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint file to load weights from")
    parser.add_argument("--vocab", type=str, default=None, help="Vocab pickle file to load one-hot encoding metrics")

    """Global Parameters (shared across modules)"""
    parser.add_argument("--node_dim", type=int)
    parser.add_argument("--edge_dim", type=int)
    parser.add_argument('--gnn_type', type=str, default='sage', choices=['sage'])  # TODO: add more choices
    parser.add_argument("--hid_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=7)
    parser.add_argument("--ablation", type=list, default=[])

    """Customized Parameters (for feature engineering)"""
    parser.add_argument("--node_drop", type=bool, default=True)
    parser.add_argument("--UV_Net", type=bool, default=True)
    parser.add_argument("--image_fingerprint", type=bool, default=False)
    parser.add_argument("--MVCNN_embedding", type=bool, default=False)
    parser.add_argument("--random_seed", type=int)
    parser.add_argument("--os", type=str)
    parser.add_argument("--single_node_prediction", type=bool, default=True)

    """Return Parser"""
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    return args


def inference(args):
    """Inference on one sample"""
    assert (
            args.checkpoint is not None
    ), "Expected the --checkpoint argument to be provided"
    assert (
            args.inference_sample is not None
    ), "Expected the --inference_sample argument to be provided"
    assert (
            args.vocab is not None
    ), "Expected the --vocab argument to be provided"

    """Recover the model from checkpoint"""
    model = ClassificationGNN.load_from_checkpoint(args.checkpoint)

    """Recover the vocab dictionary from pickle"""
    with open(args.vocab, "rb") as f:
        args.vocab = pickle.load(f)

    if args.type == "single_sample":
        print("Performing inference on a single assembly sample - all node prediction for the sample")

        assert len(os.listdir(args.inference_sample)) == 1

        """Generate the dataset - for the inference sample"""
        Dataset = AssemblyGraphs
        inference_data = Dataset(args, root_dir=args.inference_sample, split="inference")
        inference_loader = inference_data.get_dataloader(batch_size=1, shuffle=False)

        """Obtain predictions - inference on ONE sample"""
        model.eval()
        predictions, truths, all_body_ids = [], [], []
        correct, all = 0, 0

        for batch in inference_loader:
            num_nodes = batch["assembly_graph"].x.shape[0]

        for i in tqdm(range(num_nodes), desc="Inference on nodes of the sample assembly"):
            with torch.no_grad():
                for batch in inference_loader:
                    body_ids = batch["body_ids"][i]
                    preds, labels = model.inference_step(batch, i)
                    preds = preds.tolist()
                    labels = labels.tolist()

                    predictions.append(preds)
                    truths.append(labels)
                    all_body_ids.append(body_ids)

                    all += 1
                    if preds == labels:
                        correct += 1

        predictions = list(numpy.concatenate(predictions).flat)
        truths = list(numpy.concatenate(truths).flat)

        print("Body IDs:", all_body_ids)
        print("Ground Truths:", truths)
        print("Predictions:", predictions)

        print(f"Inference accuracy = {round(correct / all, 2)}")

    else:
        print("Performing inference on multiple assembly samples - one node prediction per sample")

        """Generate the dataset - for the inference sample"""
        Dataset = AssemblyGraphs
        inference_data = Dataset(args, root_dir=args.inference_sample, split="inference")
        inference_loader = inference_data.get_dataloader(batch_size=16, shuffle=False)

        """Obtain predictions - inference on ONE sample"""
        model.eval()
        predictions, truths = [], []
        correct, all = 0, 0

        for batch in tqdm(inference_loader, desc="Inference on sample assemblies"):
            preds, labels = model.test_step(batch, None)
            preds = preds.tolist()
            labels = labels.tolist()
            predictions.append(preds)
            truths.append(labels)

        predictions = list(numpy.concatenate(predictions).flat)
        truths = list(numpy.concatenate(truths).flat)

        for i in range(len(predictions)):
            all += 1
            if predictions[i] == truths[i]:
                correct += 1

        print(predictions)
        print(truths)

        print(f"Inference accuracy = {round(correct / all, 2)}")


if __name__ == "__main__":
    args = get_parser()

    """Setting Specific / Customized Arguments - Note: you need to use the EXACT setting as the checkpoint training"""
    # Model settings
    args.node_dropping = True  # dropping of "Metal_Ferrous_Steel (i.e., Default)" & "Paint" bodies
    args.UV_Net = True  # joint training with UV-Net enabled
    args.single_node_prediction = True  # batch-level randomization of mask for single node prediction task

    # Feature engineering settings
    args.image_fingerprint = False  # 2D image fingerprint as generated using ResNet
    args.MVCNN_embedding = True  # visual embeddings generated using MVCNN
    args.ablation = ["body_name", "occ_name"]

    # Random seed
    if args.random_seed is None:
        args.random_seed = random.randint(0, 99999999)
        print(f"[Note] Generated NEW random seed = {args.random_seed}")
    else:
        print(f"[Note] Using EXISTING random seed = {args.random_seed}")

    random.seed(args.random_seed)
    seed_everything(seed=args.random_seed, workers=True)

    """Begin Inference"""
    inference(args)
