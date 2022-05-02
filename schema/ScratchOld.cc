#include "Schema.h"
#include "DynamicCountsOld.h"
#include <schema/DateFeatures.h>
#include <schema/OneHotEncoding.h>
#include <schema/Text.h>
#include <chrono>
#include <memory>
#include <sstream>

using thirdai::schema::DynamicCountsOldBlock;
using thirdai::schema::Window;
using thirdai::schema::DataLoader;
using thirdai::schema::ABlockConfig;
using thirdai::schema::DateBlock;
using thirdai::schema::OneHotEncodingBlock;
using thirdai::schema::CharacterNGramBlock;
using thirdai::schema::WordNGramBlock;

int main(int argc, char* argv[])
{
    (void) argc;

    std::vector<Window> windows;
    
    for (uint32_t i = 0; i < 31; i++) {
        windows.push_back(Window(/* lag = */ 31 + i, /* size = */ 1));
    } 
    for (uint32_t i = 1; i < 13; i++) {
        windows.push_back(Window(/* lag = */ 31 + i, /* size = */ 7));
    } 
    for (uint32_t i = 1; i < 5; i++) {
        windows.push_back(Window(/* lag = */ 365 + i, /* size = */ 7));
    } 
    
    auto user_watch_rolling_feats = DynamicCountsOldBlock::Config(
        /* id_col = */ 1, 
        /* timestamp_col = */ 0, 
        /* target_col = */ -1, 
        /* window_configs = */ windows,
        /* timestamp_fmt = */ "%Y-%m-%d");
    auto movie_watch_rolling_feats = DynamicCountsOldBlock::Config(
        /* id_col = */ 2, 
        /* timestamp_col = */ 0, 
        /* target_col = */ -1, 
        /* window_configs = */ windows,
        /* timestamp_fmt = */ "%Y-%m-%d");
    
    auto date_feats = DateBlock::Config(0, "%Y-%m-%d", 7);
    auto user_id_feats = OneHotEncodingBlock::Config(1, 480189);
    auto movie_id_feats = OneHotEncodingBlock::Config(2, 17770);
    auto rating_feats = OneHotEncodingBlock::Config(3, 5);
    auto movie_title_char_feats = CharacterNGramBlock::Config(4, 3, 15000);
    auto movie_title_word_feats = WordNGramBlock::Config(4, 1, 15000);
    auto release_year_feats = OneHotEncodingBlock::Config(5, 100);
    
    std::vector<std::shared_ptr<ABlockConfig>> input_feats = {
        date_feats,
        user_id_feats,
        movie_id_feats,
        movie_title_char_feats,
        movie_title_word_feats,
        release_year_feats,
        user_watch_rolling_feats, 
        movie_watch_rolling_feats};
    std::vector<std::shared_ptr<ABlockConfig>> label_feats = {rating_feats};

    auto loader = DataLoader(input_feats, label_feats, 2048);

    std::stringstream ss;
    ss << argv[1];
    std::cout << "Reading from " << ss.str() << std::endl;
    loader.readCSV(ss.str(), ',');
    std::cout << "Will start exporting..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    loader.exportDataset();
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Done exporting! Took " << std::chrono::duration_cast<std::chrono::seconds>(end - start).count() << " seconds." << std::endl;
}