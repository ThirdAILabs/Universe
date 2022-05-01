#include "Schema.h"
#include "DynamicCountsOld.h"
#include <memory>

using thirdai::schema::DynamicCountsOldBlock;
using thirdai::schema::Window;
using thirdai::schema::DataLoader;
using thirdai::schema::ABlockConfig;

int main(int argc, char* argv[])
{
    (void) argc;
    (void) argv;

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
    

    std::vector<std::shared_ptr<ABlockConfig>> input_feats = {user_watch_rolling_feats, movie_watch_rolling_feats};
    std::vector<std::shared_ptr<ABlockConfig>> label_feats = {};

    auto loader = DataLoader(input_feats, label_feats, 2048);
    loader.readCSV("/share/data/netflix/date_sorted_data_500k.csv", ',');
    std::cout << "Will start exporting..." << std::endl;
    loader.exportDataset();
    std::cout << "Done exporting!" << std::endl;
}