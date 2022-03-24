from thirdai import data

image_metadata = data.Samples(
  location=data.locations.S3("uri/to/image_meta.csv"), 
  parser=data.parsers.Csv(column_names=True))

images = data.Samples(
  location=data.locations.S3("uri/to/images"), 
  parser=data.parsers.ImageDirectory())

photographer_metadata = data.Index(
  location=data.locations.S3("uri/to/photographer_meta.csv"), 
  parser=data.parsers.Json())

location_metadata = data.Index(
  location=data.locations.S3("uri/to/location_meta.csv"), 
  parser=data.parsers.Csv(column_names=False))


tower_1_vector = data.compose(
  [
    ImageEmbedding(source=images),
    TextEmbedding(source=image_metadata.at("description")),
    TextEmbedding(
      source=photographer_metadata
        .lookup(key="name", value=image_metadata.at("photographer"))
        .at("bio")),
  ]
)

tower_2_vector = data.compose(
  [
    WindowedCounts(
      window_configs=[],
      timestamp=image_metadata.at("timestamp"),
      identifier=image_metadata.at("location"),
      target=location_metadata
        .lookup(key=0, value=image_metadata.at("location"))
        .at(1)),
  ]
)

target_vector = data.compose(
  [
    Number(source=image_metadata.at("likes"))
  ]
)

pipeline = Pipeline(inputs=[tower_1_vector, tower_2_vector], targets=[target_vector], model=TwoTowerModel())

pipeline.run()
