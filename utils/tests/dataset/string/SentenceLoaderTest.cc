#include <string>
#include <gtest/gtest.h>

/**
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
 *          - the number of output strings equals the number of expected sentences.
 *  c. Correct handling of queues:
 *      - Open two files, stream "File 1." to the first and "File 2." to the second.
 *      - Append the same text into two different files.
 *      - Queue both files.
 *      - Make sure that every sentence except "File 1" and "File 2" gets loaded twice.
 *      - Make sure "File 1" and "File 2" are loaded eventually.
 */


static void checkOnlyWantedCharacters(std::string& str) {
    for (auto& c : str) {
        ASSERT_TRUE(c == ' ' || ('0' <= c && c <= '9') || ('a' <= c && c <= 'z'));
    }
}

static void noConsecutiveSpaces(std::string& str) {
    char last_c = 'a';
    for (auto& c : str) {
        ASSERT_FALSE(c == ' ' && last_c == ' ');
        last_c = c;
    }
}

static void 