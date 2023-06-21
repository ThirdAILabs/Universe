from thirdai.neural_db import ModelBazaar, NeuralDB, documents, qa

bazaar = ModelBazaar(base_url="url")
bazaar.fetch()
bazaar.list_model_names()


ndb = neural_db.NeuralDB(user_id="global")
ndb.from_checkpoint(bazaar.get_model_dir("ms_marco"))
ndb.add_documents([
    documents.CSV(...),
    documents.PDF(...),
    documents.DOCX(...),
])

references = ndb.search("query")

print(references[0].text)
print(references[0].id)
print(references[0].source)

ndb.associate(...)
ndb.upvote(...)
ndb.set_answerer_state(
    AnswererState(
        qa.OpenAI(key=""),
        qa.modules["Open AI for Question Answering"],
    ))
ndb.answer()