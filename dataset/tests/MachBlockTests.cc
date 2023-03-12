
#include <gtest/gtest.h>
#include <dataset/src/blocks/MachBlocks.h>
#include <dataset/src/utils/SegmentedFeatureVector.h>
#include <random>

namespace thirdai::dataset::tests {

class MachBlockTest : public testing::Test {
 public:
  /**
   * Generate random strings for categories. These are parseable as integers so
   * we can use both numeric and string MachIndex.
   */
  static std::vector<std::string> genRandomCategories() {
    std::mt19937 gen(892734);
    std::uniform_int_distribution<uint32_t> dist(0, 10);

    uint32_t num_categories = 10;
    std::vector<std::string> categories;
    for (uint32_t i = 0; i < num_categories; i++) {
      categories.push_back(std::to_string(dist(gen)));
    }
    return categories;
  }

  static void numHashesTest(const MachIndexPtr& index) {
    auto categories = genRandomCategories();
    auto block = MachBlock::make(0, index);
    for (auto& category : categories) {
      SegmentedSparseFeatureVector vec;
      vec.addFeatureSegment(block->featureDim());
      block->encodeCategory(category, /* num_categories_in_sample = */ 1, vec);
      ASSERT_EQ(vec.toBoltVector().len, index->numHashes());
    }
  }

  static void outputRangeTest(const MachIndexPtr& index) {
    auto categories = genRandomCategories();
    auto block = MachBlock::make(0, index);
    for (auto& category : categories) {
      SegmentedSparseFeatureVector vec;
      vec.addFeatureSegment(block->featureDim());
      block->encodeCategory(category, /* num_categories_in_sample = */ 1, vec);

      auto bv = vec.toBoltVector();
      for (uint32_t i = 0; i < bv.len; i++) {
        ASSERT_LT(bv.active_neurons[i], index->outputRange());
      }
    }
  }
};

TEST(MachBlockTest, TestNumericMachBlockNumHashes) {
  uint32_t output_range = 100;
  uint32_t num_hashes = 10;
  MachIndexPtr index = NumericCategoricalMachIndex::make(
      /* output_range = */ output_range, /* num_hashes = */ num_hashes);
  MachBlockTest::numHashesTest(index);
}

TEST(MachBlockTest, TestStringMachBlockNumHashes) {
  uint32_t output_range = 100;
  uint32_t num_hashes = 10;
  MachIndexPtr index = StringCategoricalMachIndex::make(
      /* output_range = */ output_range, /* num_hashes = */ num_hashes,
      /* max_elements = */ 1000);
  MachBlockTest::numHashesTest(index);
}

TEST(MachBlockTest, TestNumericMachBlockOutputRange) {
  uint32_t output_range = 100;
  uint32_t num_hashes = 10;
  MachIndexPtr index = NumericCategoricalMachIndex::make(
      /* output_range = */ output_range, /* num_hashes = */ num_hashes);
  MachBlockTest::outputRangeTest(index);
}

TEST(MachBlockTest, TestStringMachBlockOutputRange) {
  uint32_t output_range = 100;
  uint32_t num_hashes = 10;
  MachIndexPtr index = StringCategoricalMachIndex::make(
      /* output_range = */ output_range, /* num_hashes = */ num_hashes,
      /* max_elements = */ 1000);
  MachBlockTest::outputRangeTest(index);
}

}  // namespace thirdai::dataset::tests