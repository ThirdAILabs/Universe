
#pragma once

#include <archive/src/Archive.h>
#include <archive/src/List.h>
#include <archive/src/Map.h>
#include <bolt/src/nn/model/Model.h>
#include <bolt/src/NER/model/NerModel.h>
#include <bolt/src/train/trainer/Trainer.h>
#include <bolt/src/text_generation/GenerativeModel.h>
#include <data/src/Loader.h>
#include <data/src/TensorConversion.h>
#include <data/src/transformations/Pipeline.h>
#include <data/src/transformations/DyadicInterval.h>
#include <data/src/transformations/Transformation.h>
#include <dataset/src/DataSource.h>


namespace thirdai::bolt {

    class NerBoltModel final : public NerBackend {
        public:
            explicit NerBoltModel(bolt::ModelPtr model);
           
           std::vector<std::vector<uint32_t>> getTags(std::vector<std::vector<std::string>> tokens) final;

            metrics::History train(
                const dataset::DataSourcePtr& train_data, float learning_rate,
                uint32_t epochs, size_t batch_size,
                const std::vector<std::string>& train_metrics,
                const dataset::DataSourcePtr& val_data,
                const std::vector<std::string>& val_metrics
            ) final;

            ar::ConstArchivePtr toArchive() const final ;

            static std::shared_ptr<NerBackend> fromArchive(
                const ar::Archive& archive);

            void save(const std::string& filename) const;

            void save_stream(std::ostream& output_stream) const;

            static std::shared_ptr<NerBoltModel> load(const std::string& filename);

            static std::shared_ptr<NerBoltModel> load_stream(
                std::istream& input_stream);

        private:
            data::Loader getDataLoader(const dataset::DataSourcePtr& data,
                                        size_t batch_size, bool shuffle);

            data::PipelinePtr getTransformations(bool inference);


            bolt::ModelPtr _bolt_model;
            data::PipelinePtr _train_transforms;
            data::PipelinePtr _inference_transforms;
            data::OutputColumnsList _bolt_inputs;

            std::string _source_column = "source";
            std::string _target_column = "target";

            const size_t _vocab_size = 50257;
           
    };

}  // namespace thirdai::bolt