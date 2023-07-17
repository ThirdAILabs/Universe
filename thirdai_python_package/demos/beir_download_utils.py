def write_unsupervised_file(corpus, data_path):
    unsup_file = data_path + "/unsupervised.csv"
    with open(unsup_file, "w") as fw:
        header = "DOC_ID,TITLE,TEXT\n"
        fw.write(header)
        count = 0
        for key in corpus:
            title = corpus[key]["title"].replace(",", " ")
            title = title.replace("\r", " ")
            title = title.replace("\n", " ")
            text = corpus[key]["text"].replace(",", " ")
            text = text.replace("\r", " ")
            text = text.replace("\n", " ")
            fw.write(str(count) + "," + title + "," + text + "\n")
            count += 1


def remap_doc_ids(corpus):
    doc_ids_to_integers = {}
    count = 0
    for key in corpus:
        doc_ids_to_integers[key] = count
        count += 1
    return doc_ids_to_integers


def remap_query_answers(qrels, doc_ids_to_integers):
    new_qrels = {}
    for key in qrels:
        output = {}
        for doc_id in qrels[key]:
            if qrels[key][doc_id] > 0:
                output[str(doc_ids_to_integers[doc_id])] = qrels[key][doc_id]
        new_qrels[key] = output
    return new_qrels


def write_supervised_file(queries, answers, data_path, filename):
    sup_train_file = data_path + "/" + filename
    with open(sup_train_file, "w") as fw:
        fw.write("QUERY,DOC_ID\n")

        for key in queries:
            query = queries[key].replace(",", " ")
            doc_ids = ":".join(list(answers[key].keys()))
            fw.write(query + "," + doc_ids + "\n")
