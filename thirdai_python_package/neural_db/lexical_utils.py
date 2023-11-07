import random
import re
from collections import defaultdict

import nltk
from nltk.corpus import wordnet as wn

nltk.download("averaged_perceptron_tagger")
nltk.download("wordnet")


def char4(sentence):
    return [sentence[i : i + 4] for i in range(0, len(sentence) - 3, 2)]


def custom_tokenize(sentence):
    # print(sentence)
    sentence = sentence.lower()
    import re

    sentence = re.sub(r"[<>=`\-,.{}:|;/@#?!&~$\[\]()\"']+\ *", "", sentence)
    sentence = re.sub("\d+", "", sentence)
    tokens = []
    for word in sentence.split(" "):
        if len(word) > 4:
            tokens.extend(char4(word))
    # print(tokens)
    return set(tokens)


def rank_documents(query_tokens, docs_tokens):
    doc_scores = []
    for i, doc_tokens in enumerate(docs_tokens):
        doc_scores.append((i, len(query_tokens.intersection(doc_tokens))))
    sorted_scores = sorted(doc_scores, key=lambda x: x[1], reverse=True)
    return [x[0] for x in sorted_scores]


def rerank(query, search_results):
    query_tokens = custom_tokenize(query)
    docs_tokens = [custom_tokenize(r.text) for r in search_results]
    ranked_indices = rank_documents(query_tokens, docs_tokens)
    reranked_search_results = [search_results[i] for i in ranked_indices]
    return reranked_search_results


def pos_to_wordnet_pos(penntag, returnNone=False):
    morphy_tag = {"NN": wn.NOUN, "JJ": wn.ADJ, "VB": wn.VERB, "RB": wn.ADV}
    try:
        return morphy_tag[penntag[:2]]
    except:
        return None if returnNone else ""


def get_synonyms(word, pos):
    for synset in wn.synsets(word, pos=pos_to_wordnet_pos(pos)):
        for lemma in synset.lemmas():
            yield lemma.name()


def reformulate_text(text, probability=0.2, seed=42):
    random.seed(seed)
    text = re.sub(r"[^a-zA-Z0-9,.?!\s]", "", text)
    text = nltk.word_tokenize(text)
    token_list = []
    for word, tag in nltk.pos_tag(text):
        # Filter for unique synonyms not equal to word and sort.
        unique = sorted(
            set(synonym for synonym in get_synonyms(word, tag) if synonym != word)
        )
        if len(unique) < 1 or random.random() > probability:
            token_list.append(word)
        else:
            token_list.append(random.choice(unique).replace("_", " "))
    return " ".join(token_list)


def reformulate_query(query, num_candidates=25):
    expanded_queries = [query]
    for iteration in range(num_candidates):
        reformulation = reformulate_text(query, seed=iteration)
        expanded_queries.append(reformulation)
    return expanded_queries


def aggregate_search_results(preds_dict, top_k):
    output = [(k, sum(v)) for k, v in preds_dict.items()]
    output.sort(key=lambda tup: tup[1], reverse=True)
    return [x[0] for x in output[:top_k]]


def reformulate(db, query, constraints={}, top_k=100):
    expanded_queries = reformulate_query(query)
    preds_dict = defaultdict(list)
    for q in expanded_queries:
        output_results = db.search(query=q, top_k=top_k, constraints=constraints)
        for result in output_results:
            preds_dict[result].extend([result.score])
    search_results = aggregate_search_results(preds_dict, top_k=top_k)
    return search_results
