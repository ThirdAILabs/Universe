for j in range(1, 8):
  for k in range(1, 4):
    tables = 2**j 

    from thirdai import search, utils
    hf = utils.DensifiedMinHash(num_tables=tables, hashes_per_table=k, range=1000000)
    dataset = utils.load_svm("../webspam_data.svm", batch_size=10000, batches_per_load=1)
    flash = search.Flash(hf)
    flash.AddDataset(dataset)

    from timeit import default_timer as timer
    queries = utils.load_svm("../webspam_queries.svm", batch_size=100000, batches_per_load=1)
    queries.LoadNextSetOfBatches()
    start = timer()
    results = flash.QueryBatch(queries[0], top_k=100)
    end = timer()

    gt = []
    with open("../webspam_gt") as f:
      gt = [[int(i) for i in line.split()] for line in f.readlines()]


    r1_at_100 = 0
    for i, r in enumerate(results):
      r1_at_100 += gt[i][0] in r

    print("R1@100: " + str(r1_at_100 / len(results)) + ", time is " + str(end - start), flush=True)


