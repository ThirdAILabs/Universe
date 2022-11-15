#pragma once

namespace thirdai::data::python::docs {

const char* const COLUMN_BASE = R"pbdoc(
Base class for columns which represent columns of data in a dataset.
)pbdoc";

const char* const DIMENSION_INFO = R"pbdoc(
Represents the dimension and density of a column. If the column is dense then this 
is the number of elements in the column. If the column is sparse then the dimension
is the exclusive maximum index in the column.
)pbdoc";

const char* const TOKEN_COLUMN = R"pbdoc(
Constructs a TokenColumn which represents a column containing a single integer token 
per row.

Args:
    array (np.ndarray): A 1D numpy array of integers where the ith integer is used as the 
        value for the ith row. 
    dim (int): Optional parameter, indicates the dimension (exclusive max) of the values 
        in the column. This is required if this column is to be directly outputed as part
        of the dataset so that the range in the concatenated vector is known. 

Returns:
    TokenColumn:

Examples:
    >>> values = np.random.randint(low=0, high=100, size=1000)
    >>> column = data.columns.TokenColumn(array=values, dim=100)
)pbdoc";

const char* const DENSE_FEATURE_COLUMN = R"pbdoc(
Constructs a DenseFeatureColumn which represents a column containing a single float 
value per row.

Args:
    array (np.ndarray): A 1D numpy array of floats where the ith float is used as the 
        value for the ith row. 

Returns:
    DenseFeatureColumn:

Examples:
    >>> values = np.random.rand(100)
    >>> column = data.columns.DenseFeatureColumn(array=values)
)pbdoc";

const char* const STRING_COLUMN = R"pbdoc(
Constructs a StringColumn which represents a column containing a string value per 
row. 

Args:
    values (List[str]): The values for each row. 

Returns:
    StringColumn:

Examples:
    >>> values = ["a", "b", "c"]
    >>> column = data.columns.StringColumn(values=values)
)pbdoc";

const char* const TOKEN_ARRAY_COLUMN = R"pbdoc(
Constructs a TokenArrayColumn which represents a column containing a list of integers 
in each row.

Args:
    array (np.ndarray): A 2D numpy array of integers where the ith row of the array
        is used as the list of integers for the ith row of the column. 
    dim (int): Optional parameter, indicates the dimension (exclusive max) of the values 
        in the column. This is required if this column is to be directly outputed as part
        of the dataset so that the range in the concatenated vector is known. 

Returns:
    TokenArrayColumn:

Examples:
    >>> values = np.random.randint(low=0, high=100, size=(1000, 10))
    >>> column = data.columns.TokenArrayColumn(array=values, dim=100)
)pbdoc";

const char* const DENSE_ARRAY_COLUMN = R"pbdoc(
Constructs a DenseArrayColumn which represents a column containing a list of floats
in each row.

Args:
    array (np.ndarray): A 2D numpy array of floats where the ith row of the array
        is used as the ith row of the column. 

Returns:
    DenseArrayColumn:

Examples:
    >>> values = np.random.rand(100, 10)
    >>> column = data.columns.DenseFeatureColumn(array=values)
)pbdoc";

const char* const TRANSFORMATION_BASE = R"pbdoc(
The base class for transformations. Transformations represent functions that are 
applied to a `ColumnMap` to create new columns with new featurizations.
)pbdoc";

const char* const BINNING = R"pbdoc(
Constructs a Binning transformation. A binning transformation maps float values 
in the given range to integer values by constructing equal size bins within the 
range and returning the index of a bin which a value resides in. 

Args:
    input_column (str): The name of the input column to bin. This must be a DenseFeatureColumn.
    output_column (str): The name of the column that should be created with binned 
        values. This column will be a TokenColumn.
    inclusive_min (float): The minimum (inclusive) value of the column.
    exclusive_max (float): The maximum (exclusive) value of the column.

Returns:
    Binning:
)pbdoc";

const char* const STRING_HASH = R"pbdoc(
Constructs a StringHash transformation. This transformation hashes each string in 
a StringColumn to a integer value.

Args:
    input_column (str): The name of the input column to hash. This must be a StringColumn.
    output_column (str): The name of the column that should be created with hash 
        values. This column will be a TokenColumn.
    output_range (int): Optional. If supplied then the hashes will be taken mod
        output_range. This is necessary if the column will be concatenated into 
        the final output dataset.
    seed (int): Optional. The random seed to use for hashing. Default is 42.

Returns:
    StringHash:
)pbdoc";

const char* const COLUMN_PAIRGRAM = R"pbdoc(
Constructs a ColumnPairgram transformation. This transformation applies pairgrams 
to combinations of the specified input columns. 

Args:
    input_columns (List[str]): The columns to combine with pairgrams. All of these 
        columns must be TokenColumns.
    output_column (str): The name of the column that should be created with pairgram 
        values. This column will be a TokenArrayColumn.
    output_range (int): The output range of the pairgrams. 
)pbdoc";

const char* const SENTENCE_UNIGRAM = R"pbdoc(
Constructs a ColumnPairgram transformation. This transformation applies unigram 
hashes to each word in a sentence to map it to an integer value. 

Args:
    input_column (str): The name of the input text column to hash. This must be a 
        StringColumn.
    output_column (str): The name of the column that should be created with unigram 
        hash values. This column will be a TokenArrayColumn or a SparseArrayColumn 
        depending on the value of deduplicate.
    deduplicate (bool): Optional. Defaults to false. When true it will combine duplicate
        hash values that occur. This means that if you had the output hashes [1, 2, 3, 2]
        rather than returning that as a set of tokens in a TokenArrayColumn it will 
        return a SparseArrayColumn (index+value pairs) in the form [(1,1), (2,2), (3,1)].
    output_range (int): Optional. The output range of the hashes, if not specified 
        no mod will be performed on the hashes. If not specified then the resulting 
        unigrams cannot be concatenated into the final output dataset.  
)pbdoc";

const char* const TOKEN_PAIRGRAM = R"pbdoc(
Constructs a TokenPairgram transformation which applies pairgrams between all of 
the tokens of each row in a TokenArrayColumn. 

Args:
    input_column (str): The name of the input column to apply pairgrams to. This must be a 
        TokenArrayColumn.
    output_column (str): The name of the column that should be created with pairgram 
        values. This column will be a TokenArrayColumn. 
    output_range (int): The output range of the pairgrams. 
)pbdoc";

const char* const COLUMN_MAP_CLASS = R"pbdoc(
A ColumnMap represents the set of columns in the dataset. A FeaturizationPipeline
takes in a ColumnMap and outputs a ColumnMap once the transformations have been 
applied.
)pbdoc";

const char* const COLUMN_MAP_INIT = R"pbdoc(
Constructs a ColumnMap.

Args:
    columns (Dict[str, data.columns.Column]): A map from column names to the column datastructure
        representing the values of the column.

Returns:
    ColumnMap:

Examples:
    >>> column1 = data.columns.DenseFeatureColumn(...)
    >>> column2 = data.columns.TokenArrayColumn(...)
    >>> columns = data.ColumnMap({"column1": column1, "column2": column2})
)pbdoc";

const char* const COLUMN_MAP_TO_DATASET = R"pbdoc(
This method outputs a BoltDataset formed by concatenating the values of the specified
columns rowwise. If any of the columns are sparse then the output will be sparse 
vectors, otherwise the returned vectors are dense. The vectors are concatenated in
the order supplied to this method, and not all of the columns must be supplied. 
Any Token columns that are being concatenated must have a dimension, otherwise their
contribution to the output range cannot be known. StringColumns cannot be concatenated.

Args:
    columns (List[str]): The names of the columns to output.
    batch_size (int): The batch size of the dataset returned. 

Returns
    dataset.BoltDataset:

Examples:
    >>> column1 = data.columns.DenseFeatureColumn(...)
    >>> column2 = data.columns.TokenArrayColumn(...)
    >>> column3 = data.columns.TokenColumn(...)
    >>> columns = data.ColumnMap({"column1": column1, "column2": column2, "column3": column3})
    >>> dataset = columns.convert_to_dataset(["column3", "column1"], batch_size=10)
)pbdoc";

const char* const FEATURIZATION_PIPELINE_CLASS = R"pbdoc(
This class represents a collection of transformations that are applied to a ColumnMap.
Transformations are applied to the ColumnMap in the order they are specified to __init__
and this means that transformations can use the output columns of previous transformations
as their input(s) columns. 
)pbdoc";

const char* const FEATURIZATION_PIPELINE_INIT = R"pbdoc(
Constructs a FeaturizationPipeline.

Args:
    transformations (List[data.transformations.Transformation]): The transformations
        to apply in the pipeline. Note that transformations are applied in the order
        specified when passed in.

Returns:
    FeaturizationPipeline:

Examples:
    >>> pipeline = data.FeaturizationPipeline(transformations=[
      data.tranformations.Binning(...),
      data.transformations.TokenPairgram(...),
    ])
)pbdoc";

const char* const FEATURIZATION_PIPELINE_FEATURIZE = R"pbdoc(
Applies the transformations in the featurization pipeline to the given ColumnMap 
and returns a new ColumnMap with the result. The original ColumnMap is not modified 
but the data stored in the original ColumnMap and the returned ColumnMap may be 
shared using smart pointers. Essentially the map of column names to data is not 
changed, but the data can be shared between the ColumnMaps without copying.

Args:
    columns (data.ColumnMap): The column map to featurize. 

Returns
    data.ColumnMap:

Examples:
    >>> pipeline = data.FeaturizationPipeline(transformations=[
      data.tranformations.Binning(...),
      data.transformations.TokenPairgram(...),
    ])
    >>> columns = data.ColumnMap({"column1": column1, "column2": column2, "column3": column3})
    >>> new_columns = pipeline.featurize(columns)
    >>> dataset = new_columns.convert_to_dataset(...)
)pbdoc";

}  // namespace thirdai::data::python::docs