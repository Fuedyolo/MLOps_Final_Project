from tests import _PATH_DATA
import torch
import pytest
from src_2.models.train_model import main
import os.path

@pytest.mark.skipif(not os.path.exists(_PATH_DATA + '/dataset.pt'), reason="Data files not found")
def test_data():
    dataset = torch.load(_PATH_DATA + '/dataset.pt')
    assert dataset.num_classes==6, "Dataset did not have the correct number of classes"
    assert dataset[0]['x'].shape == torch.Size([3327, 3703]), "Dataset did not have the correct shape"