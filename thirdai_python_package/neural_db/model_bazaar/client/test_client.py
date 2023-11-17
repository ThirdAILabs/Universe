# %%
from bazaar_client import ModelBazaar

# %%
bazaar = ModelBazaar(base_url="http://0.0.0.0:8000/api/")
# bazaar.sign_up(email="mritunjay@thirdai.com", password="password", username="mj3ai")

# %%
bazaar.log_in(email="mritunjay@thirdai.com", password="password")

# %%
model = bazaar.train(
    "model-1",
    docs=["/Users/mjay/Documents/fast_fft.pdf"],
    is_async=True,
)

# %%
bazaar.await_train(model)

# %%
ndb_client = bazaar.deploy(
    model_identifier="mj3ai/model-1",
    deployment_name="deployment-2",
    is_async=True,
)

# %%
bazaar.await_deploy(ndb_client)

# %%
ndb_client.insert(
    files=["/Users/mjay/Documents/MACH.pdf", "/Users/mjay/Documents/OpenMPIInstall.pdf"]
)

# %%
results = ndb_client.search(query="who are the authors", top_k="5")

# %%
query_id = results["query_id"]
query_text = results["query_text"]
references = results["references"]
for reference in references:
    print(reference["text"])

# %%
ndb_client.associate(query1="authors", query2="objective")

# %%
best_answer = references[4]
ndb_client.upvote(query_id=query_id, query_text=query_text, reference=best_answer)


# %%
bazaar.undeploy(ndb_client)

# %%
bazaar.delete(model_identifier="mj3ai/model-1")

# %%
bazaar.list_models()

# %%
ndb_client = bazaar.deploy("mj3ai/model-1", deployment_name="deployment-1")

# %%
bazaar.list_deployments()

# %%
ndb_client = bazaar.connect("mj3ai/model-1:mj3ai/deployment-1")

# %%
bazaar.push_model(
    model_name="test-upload",
    local_path="model_bazaar_cache/mj3ai/test-model-4/model.ndb",
    access_level="private",
)

# %%
# Activate your thirdai license
# thirdai.licensing.activate("your-thirdai-activation-key")
ndb_model = bazaar.pull_model(model_identifier="mj3ai/model-1")

# %%
