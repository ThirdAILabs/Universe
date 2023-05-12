import os
from thirdai import bolt
from ...configs.cold_start_configs import *
from ...configs.graph_configs import *
from ...configs.mach_configs import *
from ...configs.udt_configs import *

config = YelpPolarityUDTConfig

data_types = config.get_data_types("./")
model = bolt.UniversalDeepTransformer(
    data_types=data_types,
    target=config.target,
    integer_target=config.integer_target,
    n_target_classes=config.n_target_classes,
    temporal_tracking_relationships=config.temporal_relationships,
    delimiter=config.delimiter,
    model_config=None,
    options=config.options,
)

model.save("test_udt.model")