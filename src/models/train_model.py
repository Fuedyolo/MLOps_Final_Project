import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from src.models.model import GCN
import matplotlib.pyplot as plt 
   
@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('model_filepath', type=click.Path())
@click.argument('figure_filepath', type=click.Path())
@click.option('--lr', type=float, default=0.01)
@click.option('--weight_decay', type=int, default=5e-4)
@click.option('--epochs', type=int, default=200)
def main(input_filepath, model_filepath, figure_filepath, lr, weight_decay, epochs):
    logger = logging.getLogger(__name__)
    logger.info('training model')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    dataset = torch.load(input_filepath)
    data = dataset[0].to(device)
    model = GCN(dataset.num_node_features, dataset.num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_loss = []
    model.train()
    for epoch in range(epochs):
        
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())

    torch.save(model.state_dict(), model_filepath)
    
    plt.plot(range(1, epochs + 1), train_loss)
    plt.xlabel('Training step')
    plt.ylabel('Training loss') 
    plt.savefig(figure_filepath)    
    
if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()