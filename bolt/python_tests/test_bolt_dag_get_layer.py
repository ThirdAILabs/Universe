import pytest
from utils import get_simple_concat_model

def test_get_layer():
  model = get_simple_concat_model(
      hidden_layer_top_dim=10,
      hidden_layer_bottom_dim=10,
      hidden_layer_top_sparsity=1,
      hidden_layer_bottom_sparsity=1,
      num_classes=10,
  )
  
  fc_2 = model.get_layer("fc_2")
  concat_1 = model.get_layer("concat_1")

  with pytest.raises(Exception, match=r"A node with name.*was not found"):
    model.get_layer("does_not_exist")

  