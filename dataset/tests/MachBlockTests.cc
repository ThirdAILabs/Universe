
#include <gtest/gtest.h>
#include <dataset/src/blocks/MachBlock.h>
#include <dataset/src/blocks/MachIndex.h>
#include <dataset/src/utils/SegmentedFeatureVector.h>
#include <random>

namespace thirdai::dataset::tests {

static uint32_t output_range = 100;
static uint32_t num_hashes = 10;

class MachBlockTest : public testing::Test {
 public:
  static StringCategoricalMachIndexPtr stringMachIndex() {
    return StringCategoricalMachIndex::make(
        /* output_range = */ output_range, /* num_hashes = */ num_hashes,
        /* max_elements = */ 1000);
  }

  static NumericCategoricalMachIndexPtr numericMachIndex() {
    return NumericCategoricalMachIndex::make(
        /* output_range = */ output_range, /* num_hashes = */ num_hashes,
        /* n_target_classes = */ 10);
  }

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

  static std::vector<SegmentedSparseFeatureVector> makeMachOutputVectors(
      const std::vector<std::string>& categories, const MachIndexPtr& index) {
    auto block = MachBlock::make(0, index);
    std::vector<SegmentedSparseFeatureVector> segmented_vecs;
    for (const auto& category : categories) {
      SegmentedSparseFeatureVector vec;
      vec.addFeatureSegment(block->featureDim());
      block->encodeCategory(category, /* num_categories_in_sample = */ 1, vec);
      segmented_vecs.push_back(vec);
    }
    return segmented_vecs;
  }

  static void numHashesTest(const MachIndexPtr& index) {
    auto categories = genRandomCategories();
    auto vecs = makeMachOutputVectors(categories, index);
    for (auto& vec : vecs) {
      ASSERT_EQ(vec.toBoltVector().len, index->numHashes());
    }
  }

  static void outputRangeTest(const MachIndexPtr& index) {
    auto categories = genRandomCategories();
    auto vecs = makeMachOutputVectors(categories, index);
    for (auto& vec : vecs) {
      auto bv = vec.toBoltVector();
      for (uint32_t i = 0; i < bv.len; i++) {
        ASSERT_LT(bv.active_neurons[i], index->outputRange());
      }
    }
  }

  static void compareSegmentedVecs(
      std::vector<SegmentedSparseFeatureVector>& vecs1,
      std::vector<SegmentedSparseFeatureVector>& vecs2) {
    ASSERT_EQ(vecs1.size(), vecs2.size());

    for (uint32_t vec = 0; vec < vecs1.size(); vec++) {
      auto bv1 = vecs1[vec].toBoltVector();
      auto bv2 = vecs2[vec].toBoltVector();
      ASSERT_EQ(bv1.len, bv2.len);
      for (uint32_t i = 0; i < bv1.len; i++) {
        ASSERT_EQ(bv1.activations[i], bv2.activations[i]);
        ASSERT_EQ(bv1.active_neurons[i], bv2.active_neurons[i]);
      }
    }
  }
};

TEST(MachBlockTest, TestNumericMachBlockNumHashes) {
  auto index = MachBlockTest::numericMachIndex();
  MachBlockTest::numHashesTest(index);
}

TEST(MachBlockTest, TestStringMachBlockNumHashes) {
  auto index = MachBlockTest::stringMachIndex();
  MachBlockTest::numHashesTest(index);
}

TEST(MachBlockTest, TestNumericMachBlockOutputRange) {
  auto index = MachBlockTest::numericMachIndex();
  MachBlockTest::outputRangeTest(index);
}

TEST(MachBlockTest, TestStringMachBlockOutputRange) {
  auto index = MachBlockTest::stringMachIndex();
  MachBlockTest::outputRangeTest(index);
}

TEST(MachBlockTest, TestNumericMachBlockDeterminism) {
  auto categories = MachBlockTest::genRandomCategories();
  auto index = MachBlockTest::numericMachIndex();
  auto vecs1 = MachBlockTest::makeMachOutputVectors(categories, index);
  auto vecs2 = MachBlockTest::makeMachOutputVectors(categories, index);

  MachBlockTest::compareSegmentedVecs(vecs1, vecs2);
}

TEST(MachBlockTest, TestStringMachBlockDeterminism) {
  auto categories = MachBlockTest::genRandomCategories();
  auto index = MachBlockTest::stringMachIndex();
  auto vecs1 = MachBlockTest::makeMachOutputVectors(categories, index);
  auto vecs2 = MachBlockTest::makeMachOutputVectors(categories, index);

  MachBlockTest::compareSegmentedVecs(vecs1, vecs2);
}

}  // namespace thirdai::dataset::tests
