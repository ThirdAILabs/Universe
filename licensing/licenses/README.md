# Note on license creation:

The licenses following licenses were created on 3/22/2024 with an expiration time of 1000 days later. 
- `max_output_dim_100_license`
- `max_train_samples_100_license`
- `no_save_load_license`

The following commands were used (on Blade):
- `./build/licensing/src/create_signed_license /share/keys/new_private_key.der /share/keys/new_public_key.der max_output_dim_100_license 1000`
- `./build/licensing/src/create_signed_license /share/keys/new_private_key.der /share/keys/new_public_key.der max_train_samples_100_license 1000`
- `./build/licensing/src/create_signed_license /share/keys/new_private_key.der /share/keys/new_public_key.der no_save_load_license 1000`
