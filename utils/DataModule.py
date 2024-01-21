import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import Dataset, Subset, DataLoader
from tgb.linkproppred.dataset import LinkPropPredDataset
import torch
import numpy as np
from utils.DataLoader import Data
from utils.utils import NegativeEdgeSampler


class FlightDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, negative_sample_strategy, seed: int = None):
        super().__init__()
        self.batch_size = batch_size
        self.negative_sample_strategy = negative_sample_strategy
        self.seed = seed
        self.dataset = FlightDataSet("tgbl-flight", negative_sample_strategy=self.negative_sample_strategy, seed=self.seed)
        
    
    def setup(self, stage: str):
        self.train_data = self.dataset.get_train_data()
        self.val_data = self.dataset.get_val_data()
        self.test_data = self.dataset.get_test_data()
    
    def train_dataloader(self):
        return DataLoader(dataset=self.train_data,
                          batch_size=self.batch_size,
                          shuffle=False)
    
    def val_dataloader(self):
        return DataLoader(dataset=self.val_data,
                          batch_size=self.batch_size,
                          shuffle=False)
    
    def test_dataloader(self):
        return DataLoader(dataset=self.test_data,
                          batch_size=self.batch_size,
                          shuffle=True)
        
    def get_node_features(self):
        return self.dataset.get_node_features()
    
    def get_edge_features(self):
        return self.dataset.get_edge_features()
    
    def get_full_data_as_Data(self):
        return self.dataset.get_full_data_as_Data()
    
    
class FlightDataSet(Dataset):
    def __init__(self, dataset_name: str, negative_sample_strategy: str = 'random', seed: int = None):
        super().__init__()
        self.dataset = LinkPropPredDataset(name=dataset_name, root="datasets", preprocess=True)
        self.num_nodes = len(set(self.dataset.full_data["sources"]) | set(self.dataset.full_data["destinations"]))

        self.full_data = self.dataset.full_data
        self.train_mask = self.dataset._train_mask
        self.val_mask = self.dataset._val_mask
        self.test_mask = self.dataset._test_mask

        self.negative_sample_strategy = negative_sample_strategy
        self.seed = seed
        self.earliest_timestamp = self.dataset.full_data["timestamps"][0]
        self.neg_edge_sampler = NegativeEdgeSampler(src_node_ids=self.full_data["sources"], 
                                                    dst_node_ids=self.full_data["destinations"], 
                                                    interact_times=self.full_data["timestamps"], 
                                                    negative_sample_strategy=self.negative_sample_strategy, 
                                                    seed=self.seed)
        
        print(f"the dataset has {self.full_data['sources'].shape[0]} edges and {self.num_nodes} different nodes, earliest timestamp: {self.earliest_timestamp}, edge_feat shape: {self.full_data['edge_feat'].shape}, and {len(np.unique(self.dataset.full_data['timestamps']))} steps")
            
        
    def __len__(self):
        return self.full_data.size()
    
    def __getitem__(self, index):
        src_node_id = self.full_data["sources"][index]
        dst_node_id = self.full_data["destinations"][index]
        timestamp = self.full_data["timestamps"][index]
        _, neg_dst_id = self.neg_edge_sampler.sample(size=1, batch_src_node_ids=src_node_id, batch_dst_node_ids=dst_node_id, current_batch_end_time=timestamp, current_batch_start_time=timestamp)
        return (src_node_id.astype(np.longlong), 
                dst_node_id.astype(np.longlong), 
                timestamp.astype(np.float64),
                neg_dst_id.astype(np.longlong))
    
    def get_train_data(self):
        indices = np.nonzero(self.train_mask)[0]
        indices_tensor = torch.from_numpy(indices)
        
        assert np.array_equal(self.full_data['sources'][self.train_mask].astype(np.longlong),
                              self.full_data['sources'][indices].astype(np.longlong))
        
        return Subset(self, indices_tensor)
    
    def get_val_data(self):
        indices = np.nonzero(self.val_mask)[0]
        indices_tensor = torch.from_numpy(indices)
        return Subset(self, indices_tensor)
    
    def get_test_data(self):
        indices = np.nonzero(self.test_mask)[0]
        indices_tensor = torch.from_numpy(indices)
        return Subset(self, indices_tensor)
    
    def get_full_data(self):
        return self.full_data
    
    def get_edge_features(self):
        edge_raw_features = self.full_data['edge_feat'].astype(np.float64)
        edge_raw_features = np.vstack([np.zeros(edge_raw_features.shape[1])[np.newaxis, :], edge_raw_features])

        return edge_raw_features
    
    def get_node_features(self):
        MAX_FEAT_DIM = 172
        if 'node_feat' not in self.full_data.keys():
            node_raw_features = np.zeros((self.num_nodes + 1, 1))
        else:
            node_raw_features = self.full_data['node_feat'].astype(np.float64)
            # deal with node features whose shape has only one dimension
            if len(node_raw_features.shape) == 1:
                node_raw_features = node_raw_features[:, np.newaxis]

        # add feature of padded node and padded edge
        node_raw_features = np.vstack([np.zeros(node_raw_features.shape[1])[np.newaxis, :], node_raw_features])
        return node_raw_features
    
    def get_train_data_as_Data(self):
        src_node_ids = self.full_data['sources'][self.train_mask].astype(np.longlong)
        dst_node_ids = self.full_data['destinations'][self.train_mask].astype(np.longlong)
        node_interact_times = self.full_data['timestamps'][self.train_mask].astype(np.float64)
        edge_ids = self.full_data['edge_idxs'][self.train_mask].astype(np.longlong)
        labels = self.full_data['edge_label'][self.train_mask]
        train_data = Data(src_node_ids=src_node_ids, dst_node_ids=dst_node_ids,
                      node_interact_times=node_interact_times, edge_ids=edge_ids, labels=labels)
        return train_data
    
    def get_val_data_as_Data(self):
        src_node_ids = self.full_data['sources'][self.val_mask].astype(np.longlong)
        dst_node_ids = self.full_data['destinations'][self.val_mask].astype(np.longlong)
        node_interact_times = self.full_data['timestamps'][self.val_mask].astype(np.float64)
        edge_ids = self.full_data['edge_idxs'][self.val_mask].astype(np.longlong)
        labels = self.full_data['edge_label'][self.val_mask]
        val_data = Data(src_node_ids=src_node_ids, dst_node_ids=dst_node_ids,
                      node_interact_times=node_interact_times, edge_ids=edge_ids, labels=labels)
        return val_data
    
    def get_test_data_as_Data(self):
        src_node_ids = self.full_data['sources'][self.test_mask].astype(np.longlong)
        dst_node_ids = self.full_data['destinations'][self.test_mask].astype(np.longlong)
        node_interact_times = self.full_data['timestamps'][self.test_mask].astype(np.float64)
        edge_ids = self.full_data['edge_idxs'][self.test_mask].astype(np.longlong)
        labels = self.full_data['edge_label'][self.test_mask]
        test_data = Data(src_node_ids=src_node_ids, dst_node_ids=dst_node_ids,
                      node_interact_times=node_interact_times, edge_ids=edge_ids, labels=labels)
        return test_data
    
    def get_full_data_as_Data(self):
        src_node_ids = self.full_data['sources'].astype(np.longlong)
        dst_node_ids = self.full_data['destinations'].astype(np.longlong)
        node_interact_times = self.full_data['timestamps'].astype(np.float64)
        edge_ids = self.full_data['edge_idxs'].astype(np.longlong)
        labels = self.full_data['edge_label']
        full_data = Data(src_node_ids=src_node_ids, dst_node_ids=dst_node_ids,
                      node_interact_times=node_interact_times, edge_ids=edge_ids, labels=labels)
        return full_data
    
    