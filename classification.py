import argparse
import pathlib
import time
import os
import random

import torch
import gc
import numpy
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import platform
import multiprocessing
import pickle

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

    """Command Line Arguments"""
    parser = argparse.ArgumentParser("Classification Model")
    parser.add_argument("traintest", choices=("train", "test"), default="train", help="Whether to train or test")
    parser.add_argument("--dataset_path", type=str, help="Path to dataset")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint file to load weights from", )

    """Global Parameters (populated on-run and shared across modules)"""
    parser.add_argument("--experiment_id", type=str)
    parser.add_argument("--node_dim", type=int)
    parser.add_argument("--edge_dim", type=int)
    parser.add_argument('--gnn_type', type=str, default='sage', choices=['sage'])
    parser.add_argument("--hid_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=7)
    parser.add_argument("--train_set", type=list)
    parser.add_argument("--val_set", type=list)
    parser.add_argument("--test_set", type=list)
    parser.add_argument("--vocab", type=dict)

    """Customized Parameters (for feature engineering)"""
    parser.add_argument("--node_drop", type=bool, default=True)
    parser.add_argument("--UV_Net", type=bool, default=True)
    parser.add_argument("--image_fingerprint", type=bool, default=False)
    parser.add_argument("--MVCNN_embedding", type=bool, default=True)
    parser.add_argument("--random_seed", type=int)
    parser.add_argument("--os", type=str)
    parser.add_argument("--single_node_prediction", type=bool, default=True)
    parser.add_argument("--fixed_split", type=bool, default=False)

    """Return Parser"""
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    return args


def save_results_to_csv(args, cf_acc, experiment_id, f1):
    """Organize all experimental results inside a single CSV file for easier comparison"""

    if args.os == "Windows":
        csv_dir = Path("results\\organized_results.csv")
        if not csv_dir.exists():
            df = pd.DataFrame(list(), columns=["Experiment ID", "Random Seed", "Ablation",
                                               "Micro. F1", "Confusion Acc."])
            df.to_csv("results\\organized_results.csv", index=False)

        csv_row = [experiment_id, args.random_seed, args.ablation,
                   f1, cf_acc]

        dataframe = pd.DataFrame([csv_row], columns=["Experiment ID", "Random Seed", "Ablation",
                                                     "Micro. F1", "Confusion Acc."])
        dataframe.to_csv("results\\organized_results.csv", mode='a', header=False, index=False)
    else:
        csv_dir = Path("results/organized_results.csv")
        if not csv_dir.exists():
            df = pd.DataFrame(list(), columns=["Experiment ID", "Random Seed", "Ablation",
                                               "Micro. F1", "Confusion Acc."])
            df.to_csv("results/organized_results.csv", index=False)

        csv_row = [experiment_id, args.random_seed, args.ablation,
                   f1, cf_acc]

        dataframe = pd.DataFrame([csv_row], columns=["Experiment ID", "Random Seed", "Ablation",
                                                     "Micro. F1", "Confusion Acc."])
        dataframe.to_csv("results/organized_results.csv", mode='a', header=False, index=False)


def initialization(args):
    """Initialization of the trainer & the dataset"""

    random.seed(args.random_seed)
    gc.collect()
    torch.cuda.empty_cache()

    results_path = (
        pathlib.Path(__file__).parent.joinpath("results/checkpoints")
    )
    if not results_path.exists():
        results_path.mkdir(parents=True, exist_ok=True)
        os.makedirs("results/confusion_matrices")
        os.makedirs("results/classification_reports")

    month_day = time.strftime("%m%d")
    hour_min_second = time.strftime("%H%M%S")

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=str(results_path.joinpath(month_day + '_' + hour_min_second)),
        filename="best",
        save_last=True,
    )
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=30,
        mode="min"
    )

    args.experiment_id = str(results_path.joinpath(month_day + '_' + hour_min_second))
    args.os = platform.system()

    trainer = Trainer.from_argparse_args(
        args,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=TensorBoardLogger(
            str(results_path), name=month_day + '_' + hour_min_second, version="logger"
        ),
        accumulate_grad_batches=32,
    )

    Dataset = AssemblyGraphs

    return trainer, Dataset


def train_test(args, trainer, Dataset):
    """Train & Test"""
    seed_everything(seed=args.random_seed, workers=True)

    """Obtain node and edge feature dimensions"""
    train_data = Dataset(args, root_dir=args.dataset_path, split="train")
    val_data = Dataset(args, root_dir=args.dataset_path, split="val")

    args.node_dim = train_data.node_dim()
    if args.UV_Net:
        args.node_dim += 128
    if args.single_node_prediction:
        if args.node_dropping:
            args.node_dim += 6
        else:
            args.node_dim += 8

    args.edge_dim = train_data.edge_dim()

    """Obtaining dataloaders"""
    if args.os == "Windows":
        train_loader = train_data.get_dataloader(batch_size=args.batch_size, shuffle=True)
        val_loader = val_data.get_dataloader(batch_size=args.batch_size, shuffle=False)
    else:
        train_loader = train_data.get_dataloader(batch_size=args.batch_size, shuffle=True,
                                                 num_workers=multiprocessing.cpu_count())
        val_loader = val_data.get_dataloader(batch_size=args.batch_size, shuffle=False,
                                             num_workers=multiprocessing.cpu_count())

    """Save Configurations"""
    if args.os == "Windows":
        experiment_id = args.experiment_id.split('\\')[-1]
    else:
        experiment_id = args.experiment_id.split('/')[-1]

    # Save Random Seed + Vocab + Train/val/test sets
    if args.os == "Windows":
        if not os.path.exists(f"results\\checkpoints\\{experiment_id}"):
            os.makedirs(f"results\\checkpoints\\{experiment_id}")

        with open(f"results\\checkpoints\\{experiment_id}\\random_seed.txt", "w") as f:
            f.write(str(args.random_seed))

        with open(f"results\\checkpoints\\{experiment_id}\\vocab.pickle", "wb") as f:
            pickle.dump(args.vocab, f, pickle.HIGHEST_PROTOCOL)

        with open(f"results\\checkpoints\\{experiment_id}\\train_set.txt", 'w') as f:
            for assembly in args.train_set:
                f.write(f"{assembly}\n")
        with open(f"results\\checkpoints\\{experiment_id}\\val_set.txt", 'w') as f:
            for assembly in args.val_set:
                f.write(f"{assembly}\n")

    else:
        if not os.path.exists(f"results/checkpoints/{experiment_id}"):
            os.makedirs(f"results/checkpoints/{experiment_id}")

        with open(f"results/checkpoints/{experiment_id}/random_seed.txt", "w") as f:
            f.write(str(args.random_seed))

        with open(f"results/checkpoints/{experiment_id}/vocab.pickle", "wb") as f:
            pickle.dump(args.vocab, f, pickle.HIGHEST_PROTOCOL)

        with open(f"results/checkpoints/{experiment_id}/train_set.txt", 'w') as f:
            for assembly in args.train_set:
                f.write(f"{assembly}\n")
        with open(f"results/checkpoints/{experiment_id}/val_set.txt", 'w') as f:
            for assembly in args.val_set:
                f.write(f"{assembly}\n")

    """Training"""
    if args.checkpoint:
        print("Loading from existing checkpoint - continuing previous training")
        model = ClassificationGNN.load_from_checkpoint(args.checkpoint)
    else:
        model = ClassificationGNN(args)

    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    """Testing - Making Predictions Automatically Using Best Checkpoint"""
    if args.os == "Windows":
        args.checkpoint = args.experiment_id + '\\best.ckpt'
    else:
        args.checkpoint = args.experiment_id + '/best.ckpt'

    test(args, Dataset)


def test(args, Dataset):
    """Test Only"""
    assert (
            args.checkpoint is not None
    ), "Expected the --checkpoint argument to be provided"

    test_data = Dataset(args, root_dir=args.dataset_path, split="test")
    test_loader = test_data.get_dataloader(batch_size=args.batch_size, shuffle=False)

    """Obtain node and edge feature dimensions"""
    args.node_dim = test_data.node_dim()
    if args.UV_Net:
        args.node_dim += 128
    if args.single_node_prediction:
        if args.node_dropping:
            args.node_dim += 6
        else:
            args.node_dim += 8

    args.edge_dim = test_data.edge_dim()

    model = ClassificationGNN.load_from_checkpoint(args.checkpoint)

    """Obtaining Predictions"""
    predictions, ground_truths = [], []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Inference on test loader"):
            preds, labels = model.test_step(batch, None)
            preds = preds.tolist()
            labels = labels.tolist()
            predictions.append(preds)
            ground_truths.append(labels)

    predictions = list(numpy.concatenate(predictions).flat)
    ground_truths = list(numpy.concatenate(ground_truths).flat)

    """Saving Results"""
    if args.traintest == "train":
        if args.os == "Windows":
            experiment_id = args.experiment_id.split('\\')[-1]
        else:
            experiment_id = args.experiment_id.split('/')[-1]

    # Save classification report + test set
    print(classification_report(y_pred=predictions, y_true=ground_truths))
    if args.traintest == "train":
        if args.os == "Windows":
            with open(f"results\\classification_reports\\{experiment_id}.txt", 'w') as f:
                f.write(str(classification_report(y_pred=predictions, y_true=ground_truths)))
            with open(f"results\\checkpoints\\{experiment_id}\\test_set.txt", 'w') as f:
                for assembly in args.test_set:
                    f.write(f"{assembly}\n")
        else:
            with open(f"results/classification_reports/{experiment_id}.txt", 'w') as f:
                f.write(str(classification_report(y_pred=predictions, y_true=ground_truths)))
            with open(f"results/checkpoints/{experiment_id}/test_set.txt", 'w') as f:
                for assembly in args.test_set:
                    f.write(f"{assembly}\n")
    else:
        with open(f"test_classification_report.txt", 'w') as f:
            f.write(str(classification_report(y_pred=predictions, y_true=ground_truths)))

    # Save confusion matrix
    cf = confusion_matrix(y_pred=predictions, y_true=ground_truths, normalize="true")
    cf_acc = round(sum(cf.diagonal() / cf.sum(axis=1)) / len(cf), 3)
    print("Confusion Acc = ", cf_acc)

    plt.figure(figsize=(24, 18))

    if args.node_dropping:
        label = ["Metal_Aluminum",
                 "Metal_Ferrous",
                 "Metal_Non-Ferrous",
                 "Other",
                 "Plastic",
                 "Wood"]
    else:
        label = ["Metal_Aluminum",
                 "Metal_Ferrous",
                 "Metal_Ferrous_Steel",
                 "Metal_Non-Ferrous",
                 "Other",
                 "Paint",
                 "Plastic",
                 "Wood"]

    sn.heatmap(cf, annot=True, fmt='.2f', cmap='Blues', xticklabels=label, yticklabels=label,
               annot_kws={"size": 25})
    plt.xticks(size='xx-large', rotation=45)
    plt.yticks(size='xx-large', rotation=45)
    plt.tight_layout()

    if args.traintest == "train":
        if args.os == "Windows":
            plt.savefig(fname=f'results\\confusion_matrices\\{experiment_id}.png', format='png')
            plt.savefig(fname=f'results\\confusion_matrices\\{experiment_id}.pdf', format='pdf')
        else:
            plt.savefig(fname=f'results/confusion_matrices/{experiment_id}.png', format='png')
            plt.savefig(fname=f'results/confusion_matrices/{experiment_id}.pdf', format='pdf')
    else:
        plt.savefig(fname=f'test_confusion_matrix.png', format='png')
        plt.savefig(fname=f'test_confusion_matrix.pdf', format='pdf')

    # Save organized results to CSV
    if args.traintest == "train":
        f1 = round(f1_score(y_true=ground_truths, y_pred=predictions, average='micro', zero_division=0), 3)
        save_results_to_csv(args, cf_acc, experiment_id, f1)


if __name__ == "__main__":
    args = get_parser()

    """Setting Specific / Customized Arguments"""
    # Model settings
    args.node_dropping = True  # dropping of "Metal_Ferrous_Steel (i.e., Default)" & "Paint" bodies
    args.UV_Net = True  # joint training with UV-Net enabled
    args.single_node_prediction = False  # batch-level randomization of mask for single node prediction task
    args.fixed_split = True  # use pre-defined train_val and test splits (for comparison with baseline models)

    # Feature engineering settings
    args.image_fingerprint = False  # 2D image fingerprint as generated using ResNet
    args.MVCNN_embedding = True  # visual embeddings generated using MVCNN

    # Ablations - features to remove from the node features
    ablations = [
        ["body_name", "occ_name"]
    ]

    for ablation in ablations:
        args.ablation = ablation

        # Random seed generation
        if args.random_seed is None:
            args.random_seed = random.randint(0, 99999999)
            print(f"[Note] Generated new random seed = {args.random_seed}")
        else:
            print(f"[Note] Using existing random seed = {args.random_seed}")

        """Begin The Experiment"""
        if args.traintest == "train":
            # Performing training of model, then automatically tests with best checkpoint
            trainer, Dataset = initialization(args)
            train_test(args, trainer, Dataset)
        else:
            # Performing testing of model, using existing checkpoint and random seed from experiment to be tested
            random.seed(args.random_seed)
            seed_everything(seed=args.random_seed, workers=True)

            Dataset = AssemblyGraphs
            test(args, Dataset)
