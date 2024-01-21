import torch
from utils.utils import get_neighbor_sampler
import argparse
from models.AnomalyDetector import AnomalyDetector
import pytorch_lightning as pl
from utils.DataModule import FlightDataModule
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint

parser = argparse.ArgumentParser()

parser.add_argument("-min_epochs", type=int, default=1, help="minimum number of epochs, default: 1")
parser.add_argument("-max_epochs", type=int, default=1, help="maximum number of epochs, default: 1")
parser.add_argument("-batch_size", type=int, default=20, help="batch size, default: 20")
parser.add_argument("-num_neighbors", type=int, default=20, help="number of sampled neighbors, default: 20")
parser.add_argument("-sample_neighbor_strategy", type=str, default="uniform", choices=["uniform", "recent", "time-interval-aware"], help='strategy used for neighbor sampling choices: "uniform", "recent", "time-interval-aware", default: "uniform"')
parser.add_argument("-negative_sample_strategy", type=str, default="random", choices=["random", "historical", "inductive"], help='strategy used for neighbor sampling choices: "random", "historical", "inductive", default: "random"')
parser.add_argument("-num_heads", type=int, default=4, help="number of attention heads, default: 4")
parser.add_argument("-num_layers", type=int, default=2, help="number of TGAT layers , default: 2")
parser.add_argument("-lr", type=float, default=0.001, help="learning rate, default: 0.001")
parser.add_argument("-dropout", type=float, default=0.1, help="probability of a sampled neighbor being omitted during training, default: 0.1")
parser.add_argument("-time_scaling_factor", type=float, default=0.0, help="time-scaling-factor for time-interval-aware sampling, default: 0.0")
parser.add_argument("-weight_decay", type=float, default=0.0, help="weight decay as regularization, default: 0.0")
parser.add_argument("-seed", type=int, default=42, help="random seed, default: 42")
parser.add_argument("-checkpoint", action="store_true", help="save model checkpoints")

args = parser.parse_args()


learning_rate = args.lr
weight_decay = args.weight_decay
num_layers = args.num_layers
num_heads = args.num_heads
time_scaling_factor = args.time_scaling_factor
sample_neighbor_strategy = args.sample_neighbor_strategy
negative_sample_strategy = args.negative_sample_strategy
num_neighbors = args.num_neighbors
batch_size = args.batch_size
min_epochs = args.min_epochs
max_epochs = args.max_epochs
seed = args.seed
use_checkpoint = args.checkpoint
dropout = args.dropout

time_feat_dim = 64
output_dim = 172
device = "cuda" if torch.cuda.is_available() else "cpu"
dataset_name = 'tgbl-flight'

logger = CSVLogger(save_dir='./lightning_logs', name='my_model')

if use_checkpoint:
    checkpoint_callback = ModelCheckpoint(dirpath='./model_checkpoints',
                                        filename='model-{epoch:02d}-{train_accuracy:.5f}',
                                        save_top_k=5,
                                        every_n_epochs=1,
                                        mode='max',
                                        monitor='train_accuracy')


# Initialize DataModule
dm = FlightDataModule(batch_size=batch_size, negative_sample_strategy=negative_sample_strategy, seed=seed)

print("FlightDataModule loaded")

full_data_as_Data = dm.get_full_data_as_Data()

# Initialize training neighbor sampler to retrieve temporal neighbours
train_neighbor_sampler = get_neighbor_sampler(data=full_data_as_Data, 
                                              sample_neighbor_strategy=sample_neighbor_strategy,
                                              time_scaling_factor=time_scaling_factor, 
                                              seed=1)
print("NeighborSampler initialized")

# Initialize Model
model = AnomalyDetector(node_raw_features=dm.get_node_features(), 
            edge_raw_features=dm.get_edge_features(), 
            neighbor_sampler=train_neighbor_sampler,
            time_feat_dim=time_feat_dim, 
            output_dim=output_dim, 
            num_layers=num_layers, 
            num_heads=num_heads,
            dropout=dropout, 
            device=device,
            learning_rate=learning_rate)

# Initialize Trainer
trainer = pl.Trainer(accelerator="auto", devices="auto", strategy="auto", min_epochs=min_epochs, max_epochs=max_epochs, logger=logger, callbacks=[checkpoint_callback])

# Train model
trainer.fit(model=model, datamodule=dm)
    
