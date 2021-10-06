#include "../../../dataset/string/StringDataset.h"
#include <gtest/gtest.h>
#include <fstream>
#include <iostream>

namespace thirdai::utils {

static uint64_t get_expected_num_batches(uint64_t target_batch_size,
                                         uint64_t target_batch_number,
                                         uint64_t number_of_times_loaded,
                                         uint64_t vec_num) {
  uint64_t total_batch_count =
      (vec_num + target_batch_size - 1) / target_batch_size;
  if (target_batch_number == 0) {
    return total_batch_count;
  }
  return std::min(
      target_batch_number,
      total_batch_count - ((number_of_times_loaded - 1) * target_batch_number));
}

static uint64_t get_expected_batch_size(uint64_t target_batch_size,
                                        uint64_t target_batch_number,
                                        uint64_t number_of_times_loaded,
                                        uint64_t vec_num,
                                        uint64_t batch_i_in_load) {
  // Only the last batch can have size < target_batch_size. So assume all
  // previous batches have size = target_batch_size.
  return std::min(target_batch_size, vec_num -
                                         (number_of_times_loaded - 1) *
                                             target_batch_number *
                                             target_batch_size -
                                         batch_i_in_load * target_batch_size);
}

static void evaluate_load(StringDataset& data, uint64_t target_batch_size,
                          uint64_t target_batch_number,
                          uint64_t number_of_times_loaded, uint64_t vec_num,
                          uint64_t& expected_starting_id) {
  if (target_batch_number > 0) {
    if (data.numBatches() > target_batch_number) {
      std::cout << "Num batches is greater than target batch number. Something "
                   "is terribly wrong."
                << std::endl;
    }
    ASSERT_LE(data.numBatches(), target_batch_number);
  }
  uint64_t expected_num_batches = get_expected_num_batches(
      target_batch_size, target_batch_number, number_of_times_loaded, vec_num);
  if (data.numBatches() != expected_num_batches) {
    std::cout << "Num batches expected: " << expected_num_batches
              << " got: " << data.numBatches() << std::endl
              << " Config: bn = " << target_batch_number
              << " bs = " << target_batch_size
              << " successful loads = " << number_of_times_loaded << std::endl;
  }
  ASSERT_EQ(data.numBatches(), expected_num_batches);
  for (size_t batch_i = 0; batch_i < data.numBatches(); batch_i++) {
    ASSERT_LE(data[batch_i]._batch_size, target_batch_size);
    uint64_t expected_batch_size =
        get_expected_batch_size(target_batch_size, target_batch_number,
                                number_of_times_loaded, vec_num, batch_i);
    if (data[batch_i]._batch_size != expected_batch_size) {
      std::cout << "Batch size expected: " << expected_batch_size
                << " got: " << data[batch_i]._batch_size << std::endl
                << " Config: bn = " << target_batch_number
                << " bs = " << target_batch_size
                << " successful loads = " << number_of_times_loaded
                << " batch_i = " << batch_i << std::endl;
    }
    ASSERT_EQ(data[batch_i]._batch_size, expected_batch_size);
    if (data[batch_i]._starting_id != expected_starting_id) {
      std::cout << "Batch starting id expected: " << expected_starting_id
                << " got: " << data[batch_i]._starting_id << std::endl
                << " Config: bn = " << target_batch_number
                << " bs = " << target_batch_size
                << " successful loads = " << number_of_times_loaded
                << " batch_i = " << batch_i << std::endl;
    }
    ASSERT_EQ(data[batch_i]._starting_id, expected_starting_id);
    expected_starting_id += expected_batch_size;
  }
}

static std::string filename = "FreelandSep10_2020.txt";

static void print_to_file() {
  std::string origin =
      "We brought you a lot of car plants, Michigan. We brought you a lot of "
      "car plants. You know that,"
      "right? Long time, it's been a long time since you had all these plants "
      "being built, but we brought "
      "you a lot over the last three and a half years, and we're going "
      "to bring you a lot more. "
      "That's happening. Hello, Michigan and I'm thrilled to be in Freeland "
      "with thousands of loyal,"
      "hardworking American patriots. Fifty-four days from now, we're going to "
      "win Michigan, "
      "and we're going to win four more years in the White House. This is the "
      "most important election in the history"
      "of our country. Joe Biden devoted his career to offshoring Michigan's "
      "jobs, outsourcing- "
      "We shouldn't be smiling about it, but we've changed it around. "
      "Outsourcing Michigan's factories,"
      "throwing open your borders, dragging us into endless foreign wars and "
      "surrendering our children's"
      "future to China and other far away lands. Biden supported every "
      "disastrous globalist sellout for "
      "over a half a century, including NAFTA, China, and DPP. You know that. "
      "Joe Biden surrendered your "
      "jobs to China and now he wants to surrender our country to the violent "
      "left wing mob and youre"
      "seeing that every night. If Biden wins, China wins. If Biden wins, the "
      "mob wins. If Biden wins,"
      "the rioters, anarchist, arsonist and flag burners win. But I wouldnt "
      "worry about it because"
      "hes not winning. I dont think hes winning. This is not the crowd of a "
      "person who comes in second"
      "place. You do know that. The same thing happened four years ago. It was "
      "election eve but by the time I"
      "got here, it was late. Some of you are in that audience at one oclock "
      "in the morning now election day."
      "We had 32,000 people show up and the reason I went is that I heard that "
      "crooked Hillary Clinton, "
      "Bill Clinton and Barack Hussein Obama were traveling to Michigan "
      "because they heard they had problems."
      "They were supposed to win Michigan, but they did such a lousy job that "
      "they had to come and get some votes."
      "They came and I heard about it. They said, Sir, could you do it? I "
      "hopped in the plane. We landed at "
      "12 o in the evening. I started speaking at one clock in the morning now "
      "election day."
      "32,000 people. She had 500 people. I said, Why are we going to lose "
      "Michigan? And we didnt."
      "And we didnt. First time in a long time, but now were up in the polls. "
      "We just saw polls."
      "Were up in Michigan. I dont know if thats good or bad. I dont know. I "
      "dont know if thats good "
      "or bad because in all those polls where we were down last time, we won. "
      "So maybe were better off"
      "being down a little bit. Hello, John James. I see you handsome guy. "
      "John James, get out and vote for"
      "John. Im going to introduce you in a second John James. What a job "
      "youve done. I have to tell you,"
      "he took, I saw this group of people that were running, all nice people, "
      "three very rich people and"
      "one guy that was John James and I saw them all. Right? I saw them all "
      "and I said, Wait."
      "With the wonderful invention of TiVo, one of the greats, you can play "
      "it back. I said, Who was that?"
      "He said, I dont know, sir. I said, Play it back. I said, That man is "
      "going to be a star."
      "It was John James. That was in the Republican primary. After learning "
      "about him with his incredible"
      "areer and helicopter, and hes a great, and actually a great flyer I "
      "heard the other day, a great"
      "one, a really good one as opposed to those that arent so good, but his "
      "incredible background"
      "and education at West Point and all of the things he did.";

  std::ofstream outfile(filename, std::ios::trunc);
  // std::cout << "printing" << std::endl;
  outfile << origin << std::endl;
  outfile.close();
}

TEST(StringDatasetTest, BatchesSentence) {
  // Make sure that sentence loader is working
  print_to_file();
  SentenceLoader loader;
  loader.addFileToQueue(filename);
  std::string buf;
  uint64_t expected_vec_num = 0;
  while (loader.loadNextString(buf)) {
    buf.clear();
    expected_vec_num++;
  }

  ASSERT_GT(expected_vec_num, 0);
  std::cout << "Expect " << expected_vec_num << " vectors in this dataset."
            << std::endl;

  uint64_t batch_sizes[] = {150, 100, 50, 30};
  uint64_t batch_nums[] = {150, 100, 50, 0};

  for (auto bs : batch_sizes) {
    for (auto bn : batch_nums) {
      uint64_t expected_starting_id = 0;
      StringDataset Data(FRAGMENT_TYPE::SENTENCE, bs, bn);
      Data.addFileToQueue(filename);
      size_t successful_loads = 0;

      Data.loadNextBatchSet();
      successful_loads++;
      evaluate_load(Data, bs, bn, successful_loads, expected_vec_num,
                    expected_starting_id);

      while (Data.numBatches() > 0) {
        Data.loadNextBatchSet();
        if (Data.numBatches() > 0) {
          successful_loads++;
          evaluate_load(Data, bs, bn, successful_loads, expected_vec_num,
                        expected_starting_id);
        }
      }

      // Verify that another load would lead to numbatches = 0
      Data.loadNextBatchSet();
      ASSERT_EQ(Data.numBatches(), 0);

      uint64_t expected_num_loads;

      if (bn > 0) {
        expected_num_loads = (expected_vec_num + (bs * bn) - 1) / (bs * bn);
      } else {
        expected_num_loads = 1;
      }
      if (successful_loads != expected_num_loads) {
        std::cout << "Num loads expected: " << expected_num_loads
                  << " got: " << successful_loads << std::endl
                  << " Config: bn = " << bn << " bs = " << bs << std::endl;
      }
      ASSERT_EQ(successful_loads, expected_num_loads);
    }
  }
}

}  // namespace thirdai::utils