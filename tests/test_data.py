import os.path

import pytest
import torch

data_path = 'data/processed/dataset.pt'

@pytest.mark.skipif(not os.path.exists(data_path), reason="Data files not found")
def test_data():
    dataset = torch.load(data_path)
    assert dataset.num_classes==6, "Dataset did not have the correct number of classes"
    assert dataset[0]['x'].shape == torch.Size([3327, 3703]), "Dataset did not have the correct shape"