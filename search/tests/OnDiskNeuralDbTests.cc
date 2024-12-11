#include "InvertedIndexTestUtils.h"
#include <gtest/gtest.h>
#include <search/src/neural_db/on_disk/OnDiskNeuralDb.h>
#include <filesystem>

namespace thirdai::search::ndb::tests {

class OnDiskNeuralDbTests : public ::testing::Test {
 public:
  OnDiskNeuralDbTests() { _prefix = search::tests::randomPath() + "_"; }

  void TearDown() final {
    for (const auto& db : _dbs_created) {
      std::filesystem::remove_all(db);
    }
  }

  std::string tmpDbName() {
    std::string name = _prefix + std::to_string(_dbs_created.size()) + ".db";
    _dbs_created.push_back(name);
    return name;
  }

 private:
  std::string _prefix;
  std::vector<std::string> _dbs_created;
};

void checkNdbQuery(OnDiskNeuralDB& db, const std::string& query,
                   const std::vector<ChunkId>& expected_ids) {
  auto results = db.query(query, expected_ids.size());
  ASSERT_EQ(results.size(), expected_ids.size());
  for (size_t i = 0; i < expected_ids.size(); i++) {
    ASSERT_EQ(results.at(i).first.id, expected_ids.at(i));
  }
}

void checkNdbRank(OnDiskNeuralDB& db, const std::string& query,
                  const QueryConstraints& constraints,
                  const std::vector<ChunkId>& expected_ids) {
  auto results = db.rank(query, expected_ids.size(), constraints);
  ASSERT_EQ(results.size(), expected_ids.size());
  for (size_t i = 0; i < expected_ids.size(); i++) {
    ASSERT_EQ(results.at(i).first.id, expected_ids.at(i));
  }
}

TEST_F(OnDiskNeuralDbTests, BasicRetrieval) {
  OnDiskNeuralDB db(tmpDbName());

  db.insert(
      "doc", std::nullopt,
      {"a b c d e g", "a b c d", "1 2 3", "x y z", "2 3", "c f", "f g d g",
       "c d e f", "f t q v w", "f m n o p", "f g h i", "c 7 8 9 10 11"},
      {{{"q1", MetadataValue(true)}},
       {{"q2", MetadataValue(true)}},
       {{"q1", MetadataValue(true)}},
       {},
       {{"q2", MetadataValue(true)}},
       {{"q2", MetadataValue(true)}},
       {{"q2", MetadataValue(true)}},
       {{"q1", MetadataValue(true)}, {"q2", MetadataValue(true)}},
       {},
       {},
       {},
       {{"q1", MetadataValue(true)}}});

  // Docs 2 and 1 both contain the whole query, but doc 2 is shorter so it
  // ranks higher. Docs 6 and 8 both contain "c" but 6 is shorter so the
  // query terms are more frequent within it.
  checkNdbQuery(db, {"a b c"}, {1, 0, 5, 7});
  // These candidates are a subset of the original results, plus 12 which
  // usually would score lower and not be returned, but is returned when we
  // restrict the candidates. Doc 3 is also added but scores 0.
  checkNdbRank(db, {"a b c"}, {{"q1", EqualTo::make(MetadataValue(true))}},
               {0, 7, 11});

  // Docs 7 and 11 contain the whole query, but 7 contains "g" repeated so it
  // scores higher. Docs 6, 8, 1 contain 1 term of the query. However 1 contains
  // "g" which occurs in fewer docs so it ranks higher. Between 6 and 8, 6 is
  // shorter so the query terms are more frequent within it.
  checkNdbQuery(db, {"f g"}, {6, 10, 0, 5, 7});
  // These candidates are a subset of the original results plus docs 5 & 2 which
  // score 0 are added to test they are not returned.
  checkNdbRank(db, {"f g"}, {{"q2", EqualTo::make(MetadataValue(true))}},
               {6, 5, 7});
}

}  // namespace thirdai::search::ndb::tests