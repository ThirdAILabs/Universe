#include <gtest/gtest.h>
#include <archive/src/Archive.h>
#include <data/src/transformations/StringCast.h>
#include <dataset/src/utils/SafeFileIO.h>
#include <mach/src/MachConfig.h>
#include <mach/src/MachTrainer.h>
#include <filesystem>
#include <limits>

namespace thirdai::mach::tests {

void writeSimpleDataset(const std::vector<std::string>& lines) {
  auto file = dataset::SafeFileIO::ofstream("./mach_data.csv");

  for (const auto& line : lines) {
    file << line << std::endl;
  }
}

void changeMinMaxEpochs(const std::string& path, uint32_t min_epochs,
                        uint32_t max_epochs) {
  auto input = dataset::SafeFileIO::ifstream(path);
  auto archive = ar::deserialize(input);

  auto map = ar::Map::make();
  map->set("min_epochs", ar::u64(min_epochs));
  map->set("max_epochs", ar::u64(max_epochs));

  for (const auto& [k, v] : archive->map()) {
    if (!map->contains(k)) {
      map->set(k, v);
    }
  }

  auto output = dataset::SafeFileIO::ofstream(path);
  ar::serialize(map, output);
}

TEST(MachTrainerTests, ColdStartCheckpointing) {
  if (std::filesystem::exists("mach_ckpt")) {
    std::filesystem::remove_all("./mach_ckpt");
  }

  auto mach = MachConfig()
                  .textCol("text")
                  .idCol("id")
                  .textFeatureDim(1000)
                  .embDim(50)
                  .nBuckets(100)
                  .build();

  writeSimpleDataset({
      "id,strong_col,weak_col",
      "0:10,apple,apples are a fruit",
      "1,spinach,spinach is a vegetable",
      "2:22,beans,beans are legumes",
  });

  auto data = data::TransformedIterator::make(
      std::make_shared<data::CsvIterator>("./mach_data.csv", ','),
      std::make_shared<data::StringToTokenArray>(
          "id", "id", ':', std::numeric_limits<uint32_t>::max()),
      nullptr);

  auto trainer = MachTrainer(mach, data)
                     .strongWeakCols({"strong_col"}, {"weak_col"})
                     .batchSize(2)
                     .minMaxEpochs(2, 4);

  trainer.complete("./mach_ckpt");

  ASSERT_LE(mach->model()->epochs(), 4);

  changeMinMaxEpochs("./mach_ckpt/metadata", 5, 8);

  auto loaded_trainer = MachTrainer::fromCheckpoint("./mach_ckpt");

  std::filesystem::remove_all("./mach_ckpt");

  auto mach_ckpt = loaded_trainer->complete(std::nullopt);

  ASSERT_GE(mach_ckpt->model()->epochs(), 5);
}

TEST(MachTrainerTests, TrainCheckpointing) {
  if (std::filesystem::exists("mach_ckpt")) {
    std::filesystem::remove_all("./mach_ckpt");
  }

  auto mach = MachConfig()
                  .textCol("text")
                  .idCol("id")
                  .textFeatureDim(1000)
                  .embDim(50)
                  .nBuckets(100)
                  .build();

  writeSimpleDataset({
      "id,text",
      "0:10,apples are a fruit",
      "1,spinach is a vegetable",
      "2:22,beans are legumes",
  });

  auto data = data::TransformedIterator::make(
      std::make_shared<data::CsvIterator>("./mach_data.csv", ','),
      std::make_shared<data::StringToTokenArray>(
          "id", "id", ':', std::numeric_limits<uint32_t>::max()),
      nullptr);

  auto trainer = MachTrainer(mach, data).batchSize(2).minMaxEpochs(3, 3);

  trainer.complete("./mach_ckpt");

  ASSERT_EQ(mach->model()->epochs(), 3);
  ASSERT_EQ(mach->model()->trainSteps(), 6);

  changeMinMaxEpochs("./mach_ckpt/metadata", 7, 7);

  auto loaded_trainer = MachTrainer::fromCheckpoint("./mach_ckpt");

  std::filesystem::remove_all("./mach_ckpt");

  auto mach_ckpt = loaded_trainer->complete(std::nullopt);

  ASSERT_EQ(mach_ckpt->model()->epochs(), 7);
  ASSERT_EQ(mach_ckpt->model()->trainSteps(), 14);
}

}  // namespace thirdai::mach::tests