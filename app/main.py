from fastapi import UploadFile, File
from typing import Optional
import cv2
from fastapi.responses import FileResponse


import pandas as pd
#df=pd.read_excel('Datafile.xlsx')
#df.iloc[0].values[0]
from io import BytesIO

import torch
import torch.nn.functional as F
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from src_2.models.model import GCN
import matplotlib.pyplot as plt 
import hydra
from hydra.utils import get_original_cwd
import os
import wandb
from torch_geometric.nn import GCNConv
from fastapi import FastAPI
app = FastAPI()

@app.get("/")
def read_root():
   return {"Hello": "World"}



from http import HTTPStatus

@app.get("/")
def root():
    """ Health check."""
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
    }
    return response


class GCN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super().__init__()
        self.conv1 = GCNConv(num_node_features, 16)
        self.conv2 = GCNConv(16, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)

#state_dict = torch.load('outputs/2023-01-13/22-17-30/trained_model.pt')
#print(10)

@app.post("/predict_model_v4/")
async def cv_model(efile: UploadFile = File(...)):
    dataset = torch.load(efile.file)
    data = dataset[0]

    model = GCN(dataset.num_node_features, dataset.num_classes)
    state_dict = torch.load('outputs/2023-01-13/22-17-30/trained_model.pt')
    model.load_state_dict(state_dict)
    model.eval()
    with torch.no_grad():
        pred = model(data).argmax(dim=1)
        correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
        acc = int(correct) / int(data.test_mask.sum())
        #print(f'Accuracy: {acc:.4f}')
    
    response = {
      "input": 10,
      "message": HTTPStatus.OK.phrase,
      "status-code": HTTPStatus.OK,
      "output": acc
   }
    return response

