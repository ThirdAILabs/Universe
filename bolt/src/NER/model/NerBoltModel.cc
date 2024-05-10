#include "NerBoltModel.h"

#include <cereal/archives/binary.hpp>
#include <bolt/src/nn/model/Model.h>
#include <bolt/src/train/metrics/Metric.h>
#include <bolt_vector/src/BoltVector.h>
#include <data/src/ColumnMap.h>
#include <data/src/TensorConversion.h>
#include <data/src/columns/ArrayColumns.h>
#include <data/src/transformations/NerTokenFromStringArray.h>
#include <data/src/transformations/StringCast.h>
#include <data/src/transformations/Transformation.h>
#include <dataset/src/utils/SafeFileIO.h>
#include <cmath>
#include <optional>
#include <queue>
#include <utility>
#include <vector>

namespace thirdai::bolt {
    NerBoltModel::NerBoltModel(bolt::ModelPtr model)
        : _bolt_model(std::move(model)){
            _train_transforms = getTransformations(true);
            _inference_transforms = getTransformations(false);
            _bolt_inputs = {data::OutputColumns("tokens"), data::OutputColumns("sentences")};
        }

    data::PipelinePtr NerBoltModel::getTransformations(bool inference){
        data::PipelinePtr transform;
        if(inference){
            transform = data::Pipeline::make({std::make_shared<data::NerTokenFromStringArray>(
                _source_column, "tokens", "sentences", std::nullopt
        )});
        }else{
            transform = data::Pipeline::make({std::make_shared<data::NerTokenFromStringArray>(
                _source_column, "tokens", "sentences", _target_column
            )});
        }
        transform = transform->then(std::make_shared<data::StringToTokenArray>("tokens", "tokens", ' ', _vocab_size));
        transform = transform->then(std::make_shared<data::StringToTokenArray>("sentences", "sentences", ' ', _vocab_size));
        return transform;
    }

    data::Loader NerBoltModel::getDataLoader(const dataset::DataSourcePtr& data,
                                        size_t batch_size, bool shuffle) {
        auto data_iter = data::JsonIterator::make(data, {_source_column, _target_column}, 1000);
        return data::Loader(
            data_iter, _train_transforms, nullptr, _bolt_inputs,
            {data::OutputColumns()},
            /* batch_size= */ batch_size,
            /* shuffle= */ shuffle, /* verbose= */ true,
            /* shuffle_buffer_size= */ 20000);
    }
    metrics::History NerBoltModel::train(
                const dataset::DataSourcePtr& train_data, float learning_rate,
                uint32_t epochs, size_t batch_size,
                const std::vector<std::string>& train_metrics,
                const dataset::DataSourcePtr& val_data,
                const std::vector<std::string>& val_metrics){
        

        auto train_dataset =
            getDataLoader(train_data, batch_size, /* shuffle= */ true).all();
        auto val_dataset =
            getDataLoader(val_data, batch_size, /* shuffle= */ false).all();    

        Trainer trainer(_bolt_model);

            // We cannot use train_with_dataset_loader, since it is using the older
            // dataset::DatasetLoader while dyadic model is using data::Loader
            for (uint32_t e = 0; e < epochs; e++) {
                    trainer.train_with_metric_names(
                        train_dataset, learning_rate, 1, train_metrics, val_dataset,
                        val_metrics, /* steps_per_validation= */ std::nullopt,
                        /* use_sparsity_in_validation= */ false, /* callbacks= */ {},
                        /* autotune_rehash_rebuild= */ false, /* verbose= */ true);
            }
            return trainer.getHistory();   
    }

    std::vector<std::vector<uint32_t>> NerBoltModel::getTags(std::vector<std::vector<std::string>> tokens){

        data::ColumnMap data(
            data::ColumnMap({{_source_column, data::ArrayColumn<std::string>::make(std::move(tokens), _vocab_size)}})
        );
        auto columns = _inference_transforms->applyStateless(data);
        auto tensors = data::toTensorBatches(columns, _bolt_inputs, 2048);

        std::vector<std::vector<uint32_t>> tags(tokens.size(), std::vector<uint32_t>());

        for (const auto& sub_vector : tokens) {
            std::vector<uint32_t> uint_sub_vector(sub_vector.size(), 0);
            tags.push_back(uint_sub_vector);
        }

        size_t sub_vector_index = 0;
        size_t token_index = 0;

        for (const auto& batch : tensors) {
            auto outputs = _bolt_model->forward(batch).at(0);
            
            for (size_t i=0; i<outputs->batchSize(); i+=1) {

                uint32_t predicted_tag = outputs->getVector(i).topKNeurons(1).top().second;
                // To handle empty vectos in case
                while(token_index < tags[sub_vector_index].size()){
                    sub_vector_index += 1;
                    token_index = 0;
                }
                tags[sub_vector_index][token_index] = predicted_tag;
                token_index += 1;
            }
        }

        return tags;
    }

    ar::ConstArchivePtr NerBoltModel::toArchive() const {
        auto ner_bolt_model = ar::Map::make();

        ner_bolt_model->set("bolt_model", _bolt_model->toArchive(/*with_optimizer*/ false));

        return ner_bolt_model;
    }

     std::shared_ptr<NerBoltModel> NerBoltModel::fromArchive(
        const ar::Archive& archive) {
        
        bolt::ModelPtr bolt_model = bolt::Model::fromArchive(*archive.get("bolt_model"));

        return std::make_shared<NerBoltModel>(
            NerBoltModel(bolt_model));
    }
    

    void NerBoltModel::save(const std::string& filename) const {
        std::ofstream filestream =
            dataset::SafeFileIO::ofstream(filename, std::ios::binary);
        save_stream(filestream);
    }

    void NerBoltModel::save_stream(std::ostream& output) const {
        ar::serialize(toArchive(), output);
    }

    std::shared_ptr<NerBoltModel> NerBoltModel::load(
        const std::string& filename) {
        std::ifstream filestream =
            dataset::SafeFileIO::ifstream(filename, std::ios::binary);
        return load_stream(filestream);
    }

    std::shared_ptr<NerBoltModel> NerBoltModel::load_stream(
        std::istream& input) {
        auto archive = ar::deserialize(input);
        return fromArchive(*archive);
    }
}  // namespace thirdai::bolt