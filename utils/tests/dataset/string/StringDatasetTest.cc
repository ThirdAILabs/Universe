#include "../../../dataset/string/StringDataset.h"

/**
 * First, test each string loader and each vectorizer separately.
 * LOADERS
 * 1. Sentence loader
 *  a. Correct fragmentation:
 *      - Start with a vector of sentences that only has lowercase letters, spaces and
 *      numbers. No punctuation. Stream each sentence to a file with a full stop at
 *      the end. Queue this file into the string loader. We should get a vector of strings 
 *      resembling the vector we started with.
 *  b. Correct preprocessing:
 *      - Given a text file that has uppercase letters, whitespaces other than space, and 
 *      other symbols, make sure that: 
 *          - the output strings only have spaces, numbers, and lowercase characters. 
 *          - the output strings do not have consecutive spaces
 *          - the numebr of output strings equals the number of expected sentences.
 * 
 *
 *
 * VECTORIZERS
 * 1. Trigram vectorizer
 *  Given a string, make sure that:
 *  a. all the token ids in the _indices vector are unique
 *  b. the size of _values equals the size of _indices
 *  c. the number of unique tokens = the number of unique token id's
 *  d. the count of each unique token = the count of each unique token id
 *  e. each token always hashes to the same token id
 *  f. different tokens never collide
 *  Given an empty string after, make sure that
 *  g. _indices and _values are not empty
 *
 *
 * STRING DATASET
 * Then, test String Dataset for batch quality.
 *  - Check number of loads, number of batches in each load, size of each batch
 *  - Check that the values in the batches are consistent with the output of the
 * vectorizers
 */
