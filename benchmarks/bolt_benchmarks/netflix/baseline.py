from thirdai.dataset import blocks
from model_runner import run_experiment

# FILE FORMAT
# date,user_id,movie_id,rating,movie_title,release_year

input_blocks = [
    blocks.Date(col=0), # date column
    blocks.Categorical(col=1, dim=480_189), # user id column
    blocks.Categorical(col=2, dim=17_770), # movie id column
    blocks.Text(col=4, dim=100_000), # movie title column
    blocks.Categorical(col=5, dim=100) # release year column
]

label_blocks = [
    blocks.Continuous(col=3) # rating column
]

run_experiment(input_blocks, label_blocks)

