import numpy as np
import torch
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from model import GCN

# Github

@click.command()
@click.argument('model_filepath', type=click.Path(exists=True))
@click.argument('data_filepath', type=click.Path(exists=True))

def main(model_filepath, data_filepath):
    dataset = torch.load(data_filepath)
    data = dataset[0]
    model = GCN(dataset.num_node_features, dataset.num_classes)
    state_dict = torch.load(model_filepath)
    model.load_state_dict(state_dict)
    model.eval()
    with torch.no_grad():
        pred = model(data).argmax(dim=1)
        correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
        acc = int(correct) / int(data.test_mask.sum())
        print(f'Accuracy: {acc:.4f}')
    
if __name__ == '__main__':
    main()