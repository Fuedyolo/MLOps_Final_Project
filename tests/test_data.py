#from tests import _PATH_DATA
import torch
import pytest
from src.models.train_model import main
import os.path

@pytest.mark.skipif(not os.path.exists('src/data/processed/dataset.pt'), reason="Data files not found")
def test_data():
    dataset = torch.load('src/data/processed/dataset.pt')
    assert dataset.num_classes==6, "Dataset did not have the correct number of classes"
    assert dataset[0]['x'].shape == torch.Size([3327, 3703]), "Dataset did not have the correct shape"

#def test_error_on_num_classes():
#    with pytest.raises(ValueError, match='Invalid number of classes in dataset'):
#        main()