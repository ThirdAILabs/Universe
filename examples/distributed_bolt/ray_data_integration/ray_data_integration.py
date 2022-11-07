import thirdai.distributed_bolt as db

data_parallel_ingest =  db.DataParallelIngest(dataset_type='csv')
file_locations_across_nodes = data_parallel_ingest.split_dataset_across_nodes(
                                    paths="/share/pratik/RayExperiments/RayData/Train.csv",num_workers=2
                                )
print(file_locations_across_nodes)