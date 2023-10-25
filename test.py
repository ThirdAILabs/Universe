import pandas as pd

x = [
    {
        "name": "torch",
        "input_sparsity": 1.0,
        "output_sparsity": 1.0,
        "train_time": 27.24004285625051,
        "update_time": 2.6664510135015007,
        "total_time": 29.90649567024957,
    },
    {
        "name": "bolt",
        "input_sparsity": 0.01,
        "output_sparsity": 0.01,
        "train_time": 6.360960561501997,
        "update_time": 0.7222801177485962,
        "total_time": 7.0832430154951,
    },
    {
        "name": "bolt",
        "input_sparsity": 0.01,
        "output_sparsity": 0.05,
        "train_time": 18.1161534007515,
        "update_time": 0.6530026595028176,
        "total_time": 18.769161843749316,
    },
    {
        "name": "bolt",
        "input_sparsity": 0.01,
        "output_sparsity": 0.1,
        "train_time": 39.29205532999913,
        "update_time": 0.9313946315014618,
        "total_time": 40.22345407725152,
    },
    {
        "name": "bolt",
        "input_sparsity": 0.01,
        "output_sparsity": 0.2,
        "train_time": 67.365277066001,
        "update_time": 1.4228016045053664,
        "total_time": 68.78808454150567,
    },
    {
        "name": "bolt",
        "input_sparsity": 0.05,
        "output_sparsity": 0.01,
        "train_time": 22.02724361225046,
        "update_time": 0.8014649090036983,
        "total_time": 22.828712055252254,
    },
    {
        "name": "bolt",
        "input_sparsity": 0.05,
        "output_sparsity": 0.05,
        "train_time": 72.77116461824698,
        "update_time": 1.2919524542521685,
        "total_time": 74.06312592475297,
    },
    {
        "name": "bolt",
        "input_sparsity": 0.05,
        "output_sparsity": 0.1,
        "train_time": 159.34109096050088,
        "update_time": 1.2316040354962752,
        "total_time": 160.5727022239953,
    },
]

df = pd.DataFrame.from_records(x)
df.sort_values("total_time", inplace=True)

print(df.to_markdown(index=False))
