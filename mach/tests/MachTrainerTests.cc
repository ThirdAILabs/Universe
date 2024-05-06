#include <gtest/gtest.h>
#include <data/src/ColumnMapIterator.h>
#include <data/src/transformations/StringCast.h>
#include <data/tests/MockDataSource.h>
#include <dataset/src/utils/SafeFileIO.h>
#include <mach/src/MachConfig.h>
#include <mach/src/MachTrainer.h>
#include <nlohmann/json.hpp>
#include <filesystem>
#include <limits>

using nlohmann::json;

namespace thirdai::mach::tests {
data::ColumnMapIteratorPtr makeIterator(const std::vector<std::string>& lines) {
  return data::CsvIterator::make(
      std::make_shared<data::tests::MockDataSource>(lines), ',');
}

void changeMinMaxEpochs(const std::string& path, uint32_t min_epochs,
                        uint32_t max_epochs) {
  auto input = dataset::SafeFileIO::ifstream(path);
  json metadata;
  input >> metadata;

  metadata["min_epochs"] = min_epochs;
  metadata["max_epochs"] = max_epochs;

  auto output = dataset::SafeFileIO::ofstream(path);
  output << std::setw(4) << metadata << std::endl;
}

TEST(MachTrainerTests, ColdStartCheckpointing) {
  const std::string ckpt_dir = "mach_coldstart_ckpt";

  if (std::filesystem::exists(ckpt_dir)) {
    std::filesystem::remove_all(ckpt_dir);
  }

  auto mach = MachConfig()
                  .textCol("text")
                  .idCol("id")
                  .textFeatureDim(1000)
                  .embDim(50)
                  .nBuckets(100)
                  .build();

  auto data = data::TransformedIterator::make(
      makeIterator({
          "id,strong_col,weak_col",
          "0:10,apple,apples are a fruit",
          "1,spinach,spinach is a vegetable",
          "2:22,beans,beans are legumes",
      }),
      std::make_shared<data::StringToTokenArray>(
          "id", "id", ':', std::numeric_limits<uint32_t>::max()),
      nullptr);

  auto trainer =
      MachTrainer(mach, DataCheckpoint(data, "id", {"strong_col", "weak_col"}))
          .strongWeakCols({"strong_col"}, {"weak_col"})
          .batchSize(2)
          .minMaxEpochs(2, 4);

  trainer.complete(ckpt_dir);

  ASSERT_LE(mach->model()->epochs(), 4);

  changeMinMaxEpochs(ckpt_dir + "/metadata", 5, 8);

  auto loaded_trainer = MachTrainer::fromCheckpoint(ckpt_dir);

  std::filesystem::remove_all(ckpt_dir);

  auto mach_ckpt = loaded_trainer->complete(std::nullopt);

  ASSERT_GE(mach_ckpt->model()->epochs(), 5);
}

TEST(MachTrainerTests, TrainCheckpointing) {
  const std::string ckpt_dir = "mach_train_ckpt";

  if (std::filesystem::exists(ckpt_dir)) {
    std::filesystem::remove_all(ckpt_dir);
  }

  auto mach = MachConfig()
                  .textCol("text")
                  .idCol("id")
                  .textFeatureDim(1000)
                  .embDim(50)
                  .nBuckets(100)
                  .build();

  auto data = data::TransformedIterator::make(
      makeIterator({
          "id,text",
          "0:10,apples are a fruit",
          "1,spinach is a vegetable",
          "2:22,beans are legumes",
      }),
      std::make_shared<data::StringToTokenArray>(
          "id", "id", ':', std::numeric_limits<uint32_t>::max()),
      nullptr);

  auto trainer = MachTrainer(mach, DataCheckpoint(data, "id", {"text"}))
                     .batchSize(2)
                     .minMaxEpochs(3, 3);

  trainer.complete(ckpt_dir);

  ASSERT_EQ(mach->model()->epochs(), 3);
  ASSERT_EQ(mach->model()->trainSteps(), 6);

  changeMinMaxEpochs(ckpt_dir + "/metadata", 7, 7);

  auto loaded_trainer = MachTrainer::fromCheckpoint(ckpt_dir);

  std::filesystem::remove_all(ckpt_dir);

  auto mach_ckpt = loaded_trainer->complete(std::nullopt);

  ASSERT_EQ(mach_ckpt->model()->epochs(), 7);
  ASSERT_EQ(mach_ckpt->model()->trainSteps(), 14);
}

}  // namespace thirdai::mach::tests