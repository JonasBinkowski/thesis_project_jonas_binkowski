from typing import Any
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
import torch
import torch.nn as nn
import numpy as np
from models.modules import MergeLayer, TGAT
from utils.utils import NeighborSampler
import torchmetrics


class AnomalyDetector(pl.LightningModule):
    
    def __init__(self, node_raw_features: np.ndarray, edge_raw_features: np.ndarray, neighbor_sampler: NeighborSampler,
                 time_feat_dim: int, output_dim: int = 172, num_layers: int = 2, num_heads: int = 2, dropout: float = 0.1, device: str = 'cpu', learning_rate: float = 0.001, num_neighbors: int = 20):
        super(AnomalyDetector, self).__init__()
        
        self.learning_rate = learning_rate
        self.accuracy = torchmetrics.Accuracy(task="binary", num_classes=2)
        self.f1 = torchmetrics.F1Score(task="binary", num_classes=2)
        self.auc_roc = torchmetrics.AUROC(task="binary")
        self.loss_func = nn.BCELoss()
        self.num_neighbors = num_neighbors
        self.epoch_counter = 0
        self.save_hyperparameters(ignore=['node_raw_features', 'edge_raw_features', 'neighbor_sampler'])
        
        self.tgat = TGAT(node_raw_features=node_raw_features, 
            edge_raw_features=edge_raw_features, 
            neighbor_sampler=neighbor_sampler,
            time_feat_dim=time_feat_dim, 
            output_dim=output_dim, 
            num_layers=num_layers, 
            num_heads=num_heads,
            dropout=dropout, 
            device=device)
        self.merger = MergeLayer(input_dim1=output_dim, input_dim2=output_dim, hidden_dim=output_dim, output_dim=1)
    
        
    def set_neighbor_sampler(self, neighbor_sampler: NeighborSampler):
        self.tgat.set_neighbor_sampler(neighbor_sampler=neighbor_sampler)
        
    def forward(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray,
                                                 node_interact_times: np.ndarray):
        """
        compute source and destination node temporal embeddings
        :param src_node_ids: ndarray, shape (batch_size, )
        :param dst_node_ids: ndarray, shape (batch_size, )
        :param node_interact_times: ndarray, shape (batch_size, )
        :param num_neighbors: int, number of neighbors to sample for each node
        :return:
        """
        src_node_embeddings, dst_node_embeddings = self.tgat.compute_src_dst_node_temporal_embeddings(src_node_ids=src_node_ids, dst_node_ids=dst_node_ids, node_interact_times=node_interact_times.to(torch.float64), num_neighbors=self.num_neighbors)
        
        assert src_node_embeddings.numel() % 2 == 0, "probability tensor must have even length"
        
        half_size = src_node_embeddings.size(0) // 2
        
        pos_probabilities = self.merger(input_1=src_node_embeddings[:half_size, :], input_2=dst_node_embeddings[:half_size, :]).squeeze(dim=-1).sigmoid()
        neg_probabilities = self.merger(input_1=src_node_embeddings[half_size:, :], input_2=dst_node_embeddings[half_size:, :]).squeeze(dim=-1).sigmoid()

        probabilities = torch.cat([pos_probabilities, neg_probabilities], dim=0)
        
        return probabilities
        
    def training_step(self, batch, batch_idx):
        
        batch_src_node_ids, batch_dst_node_ids, timestamps, neg_dst_node_ids = batch
        
        batch_full_src_node_ids = torch.cat([batch_src_node_ids, batch_src_node_ids], dim=0)
        batch_full_dst_node_ids = torch.cat([batch_dst_node_ids, neg_dst_node_ids.squeeze()], dim=0)
        batch_full_node_interact_times = torch.cat([timestamps, timestamps], dim=0)
        
        probabilities = self.forward(src_node_ids=batch_full_src_node_ids, dst_node_ids=batch_full_dst_node_ids, node_interact_times=batch_full_node_interact_times)
        
        
        assert probabilities.numel() % 2 == 0, "probability tensor must have even length"
        
        half_size = probabilities.size(0) // 2
        
        labels = torch.cat([torch.ones(half_size, dtype=probabilities.dtype), torch.zeros(half_size, dtype=probabilities.dtype)], dim=0).to(probabilities.device)
        
        loss = self.loss_func(probabilities, labels)
        accuracy = self.accuracy(probabilities, labels)
        f1 = self.f1(probabilities, labels)
        auroc = self.auc_roc(probabilities, labels)
        
        self.log_dict({"train_loss": loss, "train_accuracy": accuracy, "train_f1": f1, "train_auroc": auroc}, 
                      on_step=True, prog_bar=True, on_epoch=False)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        
        batch_src_node_ids, batch_dst_node_ids, timestamps, neg_dst_node_ids = batch
        
        batch_full_src_node_ids = torch.cat([batch_src_node_ids, batch_src_node_ids], dim=0)
        batch_full_dst_node_ids = torch.cat([batch_dst_node_ids, neg_dst_node_ids.squeeze()], dim=0)
        batch_full_node_interact_times = torch.cat([timestamps, timestamps], dim=0)
        
        probabilities = self.forward(src_node_ids=batch_full_src_node_ids, dst_node_ids=batch_full_dst_node_ids, node_interact_times=batch_full_node_interact_times)
        
        assert probabilities.numel() % 2 == 0, "probability tensor must have even length"
        
        half_size = probabilities.size(0) // 2
        
        labels = torch.cat([torch.ones(half_size, dtype=probabilities.dtype), torch.zeros(half_size, dtype=probabilities.dtype)], dim=0).to(probabilities.device)
        
        loss = self.loss_func(probabilities, labels)
        accuracy = self.accuracy(probabilities, labels)
        f1 = self.f1(probabilities, labels)
        auroc = self.auc_roc(probabilities, labels)
        
        self.log_dict({"val_loss": loss, "val_accuracy": accuracy, "val_f1": f1, "val_auroc": auroc}, 
                      on_step=True, prog_bar=True, on_epoch=False)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        
        batch_src_node_ids, batch_dst_node_ids, timestamps, neg_dst_node_ids = batch
        
        batch_full_src_node_ids = torch.cat([batch_src_node_ids, batch_src_node_ids], dim=0)
        batch_full_dst_node_ids = torch.cat([batch_dst_node_ids, neg_dst_node_ids.squeeze()], dim=0)
        batch_full_node_interact_times = torch.cat([timestamps, timestamps], dim=0)
        
        probabilities = self.forward(src_node_ids=batch_full_src_node_ids, dst_node_ids=batch_full_dst_node_ids, node_interact_times=batch_full_node_interact_times)
        print(f"pos: {probabilities[0]:.4f}, neg: {probabilities[-1]:.4f}")
        
        assert probabilities.numel() % 2 == 0, "probability tensor must have even length"
        
        half_size = probabilities.size(0) // 2
        
        labels = torch.cat([torch.ones(half_size, dtype=probabilities.dtype), torch.zeros(half_size, dtype=probabilities.dtype)], dim=0).to(probabilities.device)
        
        loss = self.loss_func(probabilities, labels)
        accuracy = self.accuracy(probabilities, labels)
        f1 = self.f1(probabilities, labels)
        auroc = self.auc_roc(probabilities, labels)
        
        self.log_dict({"test_loss": loss, "test_accuracy": accuracy, "test_f1": f1, "test_auroc": auroc}, 
                      on_step=True, prog_bar=True, on_epoch=False)
        
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
    
    
            
    
            