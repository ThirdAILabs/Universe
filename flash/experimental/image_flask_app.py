# This script starts a very simple flask application that indexes ImageNet
# using the penultimate layer of VGG16 as image embeddings and then exposes
# a REST API to perform ANN search. You will need to change the variables
# below the import statement for the script to work on your system. 

import tensorflow as tf
import numpy as np
from tensorflow import keras 
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.imagenet_utils import decode_predictions
import tensorflow as tf
import time
import thirdai
from flask import Flask, flash, request, send_file, abort, redirect
from PIL import Image
import io

chunk_path = "/Users/josh/IndexChunks/"
max_chunk_num_exclusive = 129
file_name_index_path = "/Users/josh/IndexChunks/all_files.npy"
average_vector_path = "/Users/josh/IndexChunks/avg.npy"
max_image_size_bytes = 16000000

reservoir_size = 500
num_tables = 500
hashes_per_table = 12
hf = thirdai.hashing.SignedRandomProjection(input_dim=4096, hashes_per_table=hashes_per_table, num_tables=num_tables)

mag_search_index = thirdai.search.MagSearch(hf, reservoir_size=reservoir_size)
num_vectors = 0
start = time.perf_counter()
num_bytes = 0
for chunk_num in range(0, max_chunk_num_exclusive):
    batch = np.load("%schunk-ave%d.npy" % (chunk_path, chunk_num))
    num_bytes += batch.nbytes
    mag_search_index.add(dense_data=batch, starting_index=num_vectors)
    num_vectors += len(batch)
end = time.perf_counter()
print(f"Loading and indexing {num_vectors} vectors ({num_bytes//1000000000}GB) took {end - start}s", flush=True)


raw_model = tf.keras.applications.VGG16(weights="imagenet")
final_model = tf.keras.Model(inputs=raw_model.input, outputs=raw_model.get_layer("fc2").output)

file_name_index = np.load(file_name_index_path)
average_vector = np.load(average_vector_path)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = max_image_size_bytes

allowed_extensions = {'jpg', 'jpeg'}
def allowed_file(filename):
  return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

image_ask_html = '''
    <!doctype html>
    <title>MagSearch Demo</title>
    <h1>MagSearch demo: upload a jpg image and get semantically similar images!</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''

@app.route("/", methods=['GET', 'POST'])
def do_ann():
  if request.method == 'POST':
    if 'file' not in request.files:
      flash("No file submitted.")
      return redirect(request.url)
    file = request.files['file']
    if not file or not allowed_file(file.filename):
      flash("Invalid file submitted.")
      return redirect(request.url)
    
    try:
      image_bytes = file.read()
      image_raw = Image.open(io.BytesIO(image_bytes))
      image_resized = image_raw.resize((224, 224))

      image_numpy = img_to_array(image_resized)
      image_batch = np.expand_dims(image_numpy, axis=0)
      image_batch_processed = tf.keras.applications.vgg16.preprocess_input(image_batch.copy())
      query_batch = final_model.predict(image_batch_processed) - average_vector

      results = mag_search_index.query(query_batch, top_k=5)

      return image_ask_html + f'''
        <br></br>
        <img src="{request.url}/get_image?index={results[0][0]}" height="250">
        <img src="{request.url}/get_image?index={results[0][1]}" height="250">
        <img src="{request.url}/get_image?index={results[0][2]}" height="250">
        <img src="{request.url}/get_image?index={results[0][3]}" height="250">
        <img src="{request.url}/get_image?index={results[0][4]}" height="250">
        '''

    except:
      flash("Internal server error, possibly corrupted image.")
      return redirect(request.url)

  return image_ask_html

@app.route('/get_image')
def get_image():
  if 'index' not in request.args:
    abort(400)
  file_id = int(request.args.get('index'))
  if file_id < 0 or file_id >= len(file_name_index):
    abort(400)
  return send_file(file_name_index[file_id], mimetype='image/jpeg')