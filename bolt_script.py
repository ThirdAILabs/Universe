from thirdai import bolt,dataset
import numpy as np
import argparse
from util import data_generator_tst

parser = argparse.ArgumentParser()
parser.add_argument("--load_factor", default=0.1, type=float)
parser.add_argument("--K", default=4, type=int)
parser.add_argument("--L", default=64, type=int)
parser.add_argument("--rp", default=12, type=int)
parser.add_argument("--rs", default=64, type=int)
parser.add_argument("--rh", default=10000, type=int)
parser.add_argument("--rb", default=50000, type=int)
args = parser.parse_args()

layers = [
    bolt.FullyConnected(dim=1024, activation_function=bolt.getActivationFunction('ReLU')), 
    bolt.FullyConnected(dim=931, activation_function=bolt.getActivationFunction('Sigmoid'), sparsity=args.load_factor,
        sampling_config=bolt.SamplingConfig(hashes_per_table=args.K, num_tables=args.L, range_pow=args.rp, reservoir_size=args.rs))
    ]

# network = bolt.Network(layers=layers, input_dim=100000)

lr = 0.001

train_data = dataset.load_bolt_svm_dataset("train_unig_big_1.svm", 2048)
# train_data_2 = dataset.load_bolt_svm_dataset("/share/data/wayfair/train_titles_pairgrams.svm", 2048)
test_file = "dev_unig_big_1.svm"
test_data = dataset.load_bolt_svm_dataset(test_file, 2048)

network.train(train_data[0],train_data[1], loss_fn=bolt.BinaryCrossEntropyLoss(), learning_rate=lr, epochs=3, rehash=args.rh, rebuild=args.rb)
network.train(train_data[0],train_data[1], loss_fn=bolt.BinaryCrossEntropyLoss(), learning_rate=lr/10, epochs=2, rehash=args.rh, rebuild=args.rb)

# network.predict(test_data, metrics=['categorical_accuracy'])

pred_probs = network.predict(test_data[0],test_data[1])[1]
top_buckets = np.argsort(-pred_probs, axis=-1)[:,:4]
top_scores = np.zeros(top_buckets.shape, dtype=np.float32)

for i in range(top_buckets.shape[0]):
    top_scores[i] = pred_probs[i,top_buckets[i]]

# top_buckets = np.argmax(pred_probs, axis=-1)

cut_off = 0.5 # 0.0000005 # 0.15 
p_1_sum = 0
r_1_sum = 0
p_sum = 0
r_sum = 0
test_count_1 = 0
test_count_2 = 0
test_data_generator = data_generator_tst([test_file], 1)

for j in range(len(top_buckets)):
    _, _, labels_batch = next(test_data_generator)
    if top_buckets[j][0] in labels_batch[0]:
        p_1_sum += 1
        r_1_sum += 1/len(labels_batch[0])
    ##
    test_count_1 += 1
    temp1 = np.where(top_scores[j]>=cut_off)[0]
    if len(temp1)>0:
        test_count_2 += 1

        s = ""
        for i in range(len(temp1)):
            s += str(top_buckets[j][temp1][i])
            if i < len(temp1) - 1:
                s += ", "
        print(s)

        temp2 = len(np.intersect1d(top_buckets[j][temp1],labels_batch[0]))
        
        p_sum += temp2/len(temp1)
        r_sum += temp2/len(labels_batch[0])
    else:
        # continue
        test_count_2 += 1
        print(top_buckets[j][0])
        temp2 = len(np.intersect1d(top_buckets[j][0],labels_batch[0]))
        p_sum += temp2
        r_sum += temp2/len(labels_batch[0])

p1_sum = p_1_sum/test_count_1
r1_sum = r_1_sum/test_count_1
f_1 = (2*p1_sum*r1_sum)/(p1_sum+r1_sum)

print('p_1: '+str(p1_sum))
print('r_1: '+str(r1_sum))
print('f_1: '+str(f_1))

p_thres_sum = p_sum/test_count_1
r_thres_sum = r_sum/test_count_1
f = (2*p_thres_sum*r_thres_sum)/(p_thres_sum+r_thres_sum)

print('p: '+str(p_thres_sum))
print('r: '+str(r_thres_sum))
print('f: '+str(f))


'''
for i in range(len(top_buckets)):
    # x_idxs_batch, x_vals_batch, labels_batch = next(test_data_generator)
    # top_buckets = network.predict(np.array(x_idxs_batch)[:,1].astype(np.uint32), np.array(x_vals_batch).astype(np.float32), np.array([0,1]).astype(np.uint32), np.array(1).astype(np.float32), np.array([0,1]).astype(np.uint32), batch_size=1)
    _, _, labels_batch = next(test_data_generator)
    # temp.append(labels_batch[0])
    # ##
    if top_buckets[i,0] in labels_batch[0]:
        p_1_sum += 1
        r_1_sum += 1/len(labels_batch[0])
    ##
    # p_cheat_sum += len(np.intersect1d(top_buckets[i][:len(labels_batch[0])],labels_batch[0]))/len(labels_batch[0])
    test_count += 1


print('p_1: '+str(p_1_sum/test_count))
print('r_1: '+str(r_1_sum/test_count))
# print('p_cheat: '+str(p_cheat_sum/test_count))

'''
