import pytorch_lightning as pl
import torchmetrics
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import random
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
from datetime import datetime

import utils.models.encoders as encoders


###############################################################################
# Helpers
###############################################################################


class _NonLinearClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, dropout=0.3):
        """
        A 3-layer MLP with linear outputs

        Args:
            input_dim (int): Dimension of the input tensor 
            num_classes (int): Dimension of the output logits
            dropout (float, optional): Dropout used after each linear layer. Defaults to 0.3.
        """
        super().__init__()
        self.linear1 = nn.Linear(input_dim, 512, bias=False)
        self.bn1 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(512, 256, bias=False)
        self.bn2 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=dropout)
        self.linear3 = nn.Linear(256, num_classes)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, inp):
        """
        Forward pass

        Args:
            inp (torch.tensor): Inputs features to be mapped to logits
                                (batch_size x input_dim)

        Returns:
            torch.tensor: Logits (batch_size x num_classes)
        """
        x = F.relu(self.bn1(self.linear1(inp)))
        x = self.dp1(x)
        x = F.relu(self.bn2(self.linear2(x)))
        x = self.dp2(x)
        x = self.linear3(x)
        return x


def concat_masked_material_features(args, batch, device, idx=None):
    """
        Create randomized masking and concatenation of material features - per batch
    """
    if args.node_dropping:
        materials_feature = F.one_hot(batch["labels"], 6)  # one-hot material vector
    else:
        materials_feature = F.one_hot(batch["labels"], 8)  # one-hot material vector

    # for each graph in batch.batch: concat one-hot mask to each batch.x, except for one random one
    random_mask = []
    for i in range(batch["assembly_graph"].num_graphs):
        mask = np.ones(batch["assembly_graph"].ptr[i + 1] - batch["assembly_graph"].ptr[i])
        # randomly choose node to mask (serve as prediction target) -> seed based on time
        if idx:
            mask[idx] = 0
        else:
            mask[random.Random(datetime.now()).randint(0, len(mask) - 1)] = 0
        random_mask.extend(mask.tolist())

    # set materials_feature to zeros if masked
    materials_feature = materials_feature.cpu() * np.array(random_mask)[:, None]
    # standardize the materials_feature
    materials_feature = torch.from_numpy(StandardScaler().fit_transform(materials_feature))

    batch["assembly_graph"].x = torch.cat((batch["assembly_graph"].x, materials_feature.to(device)),
                                          dim=-1)  # add materials_feature to batch.x

    return batch, random_mask


def mask_filter(predictions, ground_truths, mask):
    # only consider the target nodes (without ground truths material information in node features)
    mask = np.array(mask)
    predictions = predictions[mask == 0.0]
    ground_truths = ground_truths[mask == 0]
    return predictions, ground_truths


###############################################################################
# Classification model - Assembly GNN
###############################################################################


class GNNClassifier(nn.Module):
    def __init__(self, node_dim, edge_dim, hid_dim, num_materials, num_layers, network="sage"):
        super().__init__()
        self.gnn_level1 = encoders.GraphNN(node_dim, edge_dim, hid_dim, 0, num_materials, num_layers, network)

    def forward(self, x, edge_index, e):
        material_predictions = self.gnn_level1(x, edge_index, e, 0)
        return material_predictions


class ClassificationGNN(pl.LightningModule):
    def __init__(self, args):
        """
        Args:
            num_classes (int): Number of material classes that we want to predict on
        """
        super().__init__()
        self.save_hyperparameters()
        if args.node_dropping:
            self.GNN_model = GNNClassifier(args.node_dim, args.edge_dim, args.hid_dim, 6,
                                           args.num_layers, args.gnn_type)
            self.UV_model = UVNetClassifier(6)
        else:
            self.GNN_model = GNNClassifier(args.node_dim, args.edge_dim, args.hid_dim, 8,
                                           args.num_layers, args.gnn_type)
            self.UV_model = UVNetClassifier(8)
        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()
        self.test_acc = torchmetrics.Accuracy()
        self.args = args

    def forward(self, batched_assembly_graphs):
        """
            Input:
                [batched_assembly_graphs] batched assembly graphs
        """
        logits = self.GNN_model(batched_assembly_graphs.x.float(),
                                batched_assembly_graphs.edge_index,
                                batched_assembly_graphs.e.float())
        return logits

    def training_step(self, batch, batch_idx):

        labels = batch["labels"]

        if self.args.UV_Net:
            batch = self.UV_embedding_insertion(batch)

        if self.args.single_node_prediction:
            batch, random_mask = concat_masked_material_features(self.args, batch, self.device)

        batched_assembly_graphs = batch["assembly_graph"].to(self.device)
        material_predictions = self.forward(batched_assembly_graphs)

        if self.args.single_node_prediction:
            material_predictions, labels = mask_filter(material_predictions, labels, random_mask)

        loss = F.cross_entropy(material_predictions, labels, reduction="mean", weight=batch["weights"])
        self.log("train_loss", loss, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)
        preds = F.softmax(material_predictions, dim=-1)
        self.log("train_acc", self.val_acc(preds, labels), on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):

        labels = batch["labels"]

        if self.args.UV_Net:
            batch = self.UV_embedding_insertion(batch)

        if self.args.single_node_prediction:
            batch, random_mask = concat_masked_material_features(self.args, batch, self.device)

        batched_assembly_graphs = batch["assembly_graph"].to(self.device)
        material_predictions = self.forward(batched_assembly_graphs)

        if self.args.single_node_prediction:
            material_predictions, labels = mask_filter(material_predictions, labels, random_mask)

        loss = F.cross_entropy(material_predictions, labels, reduction="mean")
        self.log("val_loss", loss, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)
        preds = F.softmax(material_predictions, dim=-1)
        self.log("val_acc", self.val_acc(preds, labels), on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        labels = batch["labels"]

        if self.args.UV_Net:
            batch = self.UV_embedding_insertion(batch)

        if self.args.single_node_prediction:
            batch, random_mask = concat_masked_material_features(self.args, batch, self.device)

        batched_assembly_graphs = batch["assembly_graph"].to(self.device)
        material_predictions = self.forward(batched_assembly_graphs)

        if self.args.single_node_prediction:
            material_predictions, labels = mask_filter(material_predictions, labels, random_mask)

        loss = F.cross_entropy(material_predictions, labels, reduction="mean")
        self.log("test_loss", loss, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)
        preds = F.softmax(material_predictions, dim=-1)
        self.log("test_acc", self.val_acc(preds, labels), on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)

        preds = torch.argmax(preds, dim=-1)
        return preds, labels

    def inference_step(self, batch, idx):
        labels = batch["labels"]

        if self.args.UV_Net:
            batch = self.UV_embedding_insertion(batch)

        if self.args.single_node_prediction:
            batch, random_mask = concat_masked_material_features(self.args, batch, self.device, idx)

        batched_assembly_graphs = batch["assembly_graph"].to(self.device)
        material_predictions = self.forward(batched_assembly_graphs)

        if self.args.single_node_prediction:
            material_predictions, labels = mask_filter(material_predictions, labels, random_mask)

        preds = F.softmax(material_predictions, dim=-1)
        preds = torch.argmax(preds, dim=-1)

        return preds, labels

    def UV_embedding_insertion(self, batch):

        """Obtain embeddings from UV-Net, and the mask for mapping to the GNN nodes"""
        inputs = batch["body_graphs"].to(self.device)

        # Permutations of dimensions to become compatible with model
        inputs.ndata["x"] = inputs.ndata["x"].permute(0, 3, 1, 2)
        inputs.edata["x"] = inputs.edata["x"].permute(0, 2, 1)

        _, uv_embeddings = self.UV_model(inputs)  # Obtain embeddings from UV_model

        """Insert the UV embeddings back into the GNN node features"""
        body_embeddings = torch.zeros((batch["assembly_graph"].x.shape[0], 128)).to(self.device)
        body_embeddings[batch["mask"], :] = uv_embeddings
        batch["assembly_graph"].x = torch.cat((batch["assembly_graph"].x, body_embeddings), dim=-1)

        return batch

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters())

        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": CosineAnnealingLR(opt, T_max=10),
                "interval": 'step',
                "frequency": 1,
            }
        }


###############################################################################
# Classification model - UV Net
###############################################################################

class UVNetClassifier(nn.Module):
    """
    UV-Net solid classification model
    """

    def __init__(
            self,
            num_classes,
            crv_emb_dim=64,
            srf_emb_dim=64,
            graph_emb_dim=128,
            dropout=0.3,
    ):
        """
        Initialize the UV-Net solid classification model
        
        Args:
            num_classes (int): Number of classes to output
            crv_emb_dim (int, optional): Embedding dimension for the 1D edge UV-grids. Defaults to 64.
            srf_emb_dim (int, optional): Embedding dimension for the 2D face UV-grids. Defaults to 64.
            graph_emb_dim (int, optional): Embedding dimension for the graph. Defaults to 128.
            dropout (float, optional): Dropout for the final non-linear classifier. Defaults to 0.3.
        """
        super().__init__()
        self.curv_encoder = encoders.UVNetCurveEncoder(
            in_channels=6, output_dims=crv_emb_dim
        )
        self.surf_encoder = encoders.UVNetSurfaceEncoder(
            in_channels=7, output_dims=srf_emb_dim
        )
        self.graph_encoder = encoders.UVNetGraphEncoder(
            srf_emb_dim, crv_emb_dim, graph_emb_dim,
        )
        self.clf = _NonLinearClassifier(graph_emb_dim, num_classes, dropout)

    def forward(self, batched_graph):
        """
        Forward pass

        Args:
            batched_graph (dgl.Graph): A batched DGL graph containing the face 2D UV-grids in node features
                                       (ndata['x']) and 1D edge UV-grids in the edge features (edata['x']).

        Returns:
            torch.tensor: Logits (batch_size x num_classes)
        """
        # Input features
        input_crv_feat = batched_graph.edata["x"]
        input_srf_feat = batched_graph.ndata["x"]
        # Compute hidden edge and face features
        hidden_crv_feat = self.curv_encoder(input_crv_feat)
        hidden_srf_feat = self.surf_encoder(input_srf_feat)
        # Message pass and compute per-face(node) and global embeddings
        # Per-face embeddings are ignored during solid classification
        _, graph_emb = self.graph_encoder(
            batched_graph, hidden_srf_feat, hidden_crv_feat
        )
        # Map to logits
        out = self.clf(graph_emb)

        return out, graph_emb

    def embeddings(self, batched_graph):
        """
        Obtain graph embeddings (just like previous "forward" function, but returning "graph_emb" instead of "out")

        Args:
            batched_graph (dgl.Graph): A batched DGL graph containing the face 2D UV-grids in node features
                                       (ndata['x']) and 1D edge UV-grids in the edge features (edata['x']).

        Returns:
            torch.tensor: Graph/Node Embeddings (batch_size x graph_emb_dim)
        """
        # Input features
        input_crv_feat = batched_graph.edata["x"]
        input_srf_feat = batched_graph.ndata["x"]
        # Compute hidden edge and face features
        hidden_crv_feat = self.curv_encoder(input_crv_feat)
        hidden_srf_feat = self.surf_encoder(input_srf_feat)
        # Message pass and compute per-face(node) and global embeddings
        # Per-face embeddings are ignored during solid classification
        _, graph_emb = self.graph_encoder(
            batched_graph, hidden_srf_feat, hidden_crv_feat
        )

        return graph_emb
