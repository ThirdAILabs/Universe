import xgboost as xgb
import pandas as pd
import numpy as np

TRAIN_FILE = "./build/brazilian_houses_train.csv"
TEST_FILE = "./build/brazilian_houses_test.csv"

# model = bolt.UniversalDeepTransformer(
#     data_types={
#         "area": bolt.types.numerical(range=(11, 46350)),
#         "rooms": bolt.types.categorical(),
#         "bathroom": bolt.types.categorical(),
#         "parking_spaces": bolt.types.categorical(),
#         "hoa_(BRL)": bolt.types.numerical(range=(0, 1117000)),
#         "rent_amount_(BRL)": bolt.types.numerical(range=(450, 45000)),
#         "property_tax_(BRL)": bolt.types.numerical(range=(0, 313700)),
#         "fire_insurance_(BRL)": bolt.types.numerical(range=(3, 677)),
#         "totalBRL": bolt.types.numerical(range=(6, 14)),
#     },
#     target="totalBRL",
# )

train_df = pd.read_csv(TRAIN_FILE)
train_x, train_y = train_df.iloc[:, :-1], train_df.iloc[:, -1]

model = xgb.XGBRegressor(n_estimators=100, max_depth=6)
model = model.fit(train_x, train_y)


test_df = pd.read_csv(TEST_FILE)
test_x, test_y = test_df.iloc[:, :-1], test_df.iloc[:, -1]

y_pred = model.predict(test_x)

print(np.sqrt(np.sum(np.square(y_pred - test_y.to_numpy()))))
print(np.mean(np.abs(y_pred - test_y.to_numpy())))