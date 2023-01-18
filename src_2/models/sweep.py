import torch
import torch.nn.functional as F
from src_2.models.model import GCN
import wandb
import numpy as np

sweep_configuration = {
    'method': 'bayes',
    'name': 'sweep',
    'metric': {'goal': 'maximize', 'name': 'val_acc'},
    'parameters': 
    {
        'weight_decay': {'min': 1e-8, 'max': 1e-4},
        'epochs': {'values': [25, 50, 75, 100]},
        'lr': {'min': 0.0001, 'max': 0.1}
     }
}

sweep_id = wandb.sweep(sweep=sweep_configuration, project='MLOps')

def train_one_epoch(optimizer, model, data):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    loss = loss.item()
    pred = out.argmax(dim=1)
    correct = (pred[data.train_mask] == data.y[data.train_mask]).sum()
    acc = int(correct) / int(data.train_mask.sum())
    return acc, loss

def evaluate_one_epoch(model, data):
    model.eval()
    with torch.no_grad():
        out = model(data)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask]).item()
        pred = out.argmax(dim=1)
        correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
        acc = int(correct) / int(data.test_mask.sum())
        return acc, loss

def main():    
    run = wandb.init(project='MLOps')
    lr  =  wandb.config.lr
    weight_decay = wandb.config.weight_decay
    epochs = wandb.config.epochs
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    input_filepath = 'data/processed/dataset.pt'
    dataset = torch.load(input_filepath)
    data = dataset[0].to(device)

    model = GCN(dataset.num_node_features, dataset.num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in np.arange(1, epochs):
        train_acc, train_loss = train_one_epoch(optimizer, model, data)
        val_acc, val_loss = evaluate_one_epoch(model, data)

        wandb.log({
        'epoch': epoch, 
        'train_acc': train_acc,
        'train_loss': train_loss, 
        'val_acc': val_acc, 
        'val_loss': val_loss
      }) 

wandb.agent(sweep_id, function=main, count=4)