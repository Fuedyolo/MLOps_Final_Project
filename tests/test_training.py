import torch
from torch import nn, optim
from src_2.models.model import GCN
from torch_geometric.data import Data

def test_training():
    '''
    Check for parameter update during training
    Parameter is deemed updated if the gradient is not None and has non-zero value
    '''
    num_node_features = 10
    num_classes = 2
    num_edges = 10
    num_nodes = 10
    x = torch.randn(num_nodes, num_node_features)
    edge_index = torch.randint(0, num_classes, (2, num_edges))
    data = Data(x, edge_index)
    targets = torch.randint(0, num_classes, (num_nodes,)).type(torch.LongTensor)
    model = GCN(num_node_features, num_classes).float()
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    optimizer.zero_grad()
    log_ps = model(data)
    loss = criterion(log_ps, targets)
    loss.backward()
    optimizer.step()
    assert all((param.grad is not None) and (not torch.all(param.grad == 0)) for param in model.parameters())