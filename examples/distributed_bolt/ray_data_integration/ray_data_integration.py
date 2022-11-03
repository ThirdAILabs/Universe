import thirdai.distributed_bolt as db

data_parallel_ingest = DataParallelIngest(dataset_type='csv')
file_locations_across_nodes = data_parallel_ingest.split_dataset_across_nodes(
                                    paths='/share/pratik/RayExperiments/RayData/0/training_file.csv',num_workers=2
                                )