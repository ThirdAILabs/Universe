#include "Schema.h"
#include "DynamicCounts.h"
#include <chrono>
#include <memory>
#include <sstream>

using thirdai::schema::DynamicCountsBlock;
using thirdai::schema::Window;
using thirdai::schema::DataLoader;
using thirdai::schema::ABlockConfig;

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
    
    auto user_watch_rolling_feats = DynamicCountsBlock::Config(
        /* id_col = */ 1, 
        /* timestamp_col = */ 0, 
        /* target_col = */ -1, 
        /* window_configs = */ windows,
        /* timestamp_fmt = */ "%Y-%m-%d");
    auto movie_watch_rolling_feats = DynamicCountsBlock::Config(
        /* id_col = */ 2, 
        /* timestamp_col = */ 0, 
        /* target_col = */ -1, 
        /* window_configs = */ windows,
        /* timestamp_fmt = */ "%Y-%m-%d");
    
    date_feats = schema.Date(col=0, timestamp_fmt="%Y-%m-%d", n_years=7)
    input_blocks.append(date_feats)
    user_id_feats = schema.OneHotEncoding(col=1, out_dim=480_189)
    input_blocks.append(user_id_feats)
    movie_id_feats = schema.OneHotEncoding(col=2, out_dim=17_770)
    input_blocks.append(movie_id_feats)
    rating_feats = schema.OneHotEncoding(col=3, out_dim=5)
    label_blocks.append(rating_feats)
    movie_title_char_feats = schema.CharacterNGram(col=4, k=3, out_dim=15_000)
    input_blocks.append(movie_title_char_feats)
    movie_title_word_feats = schema.WordNGram(col=4, k=1, out_dim=15_000)
    input_blocks.append(movie_title_word_feats)
    release_year_feats = schema.OneHotEncoding(col=5, out_dim=100)
    input_blocks.append(release_year_feats)

    std::vector<std::shared_ptr<ABlockConfig>> input_feats = {user_watch_rolling_feats, movie_watch_rolling_feats};
    std::vector<std::shared_ptr<ABlockConfig>> label_feats = {};

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