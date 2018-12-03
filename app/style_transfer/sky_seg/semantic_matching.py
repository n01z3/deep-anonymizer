import csv
import gensim
from gensim.models.wrappers import FastText
import pickle
COMPUTE_WORD2VEC = 1
filename = 'semantic_data.pkl'
rel_filename = 'semantic_rel.npy'

if COMPUTE_WORD2VEC:
    class_info = 'data/edited_object150_info.csv'
    class_name = []
    with open(class_info, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for i,row in enumerate(spamreader):
            if i>0:
                name = row[-1]
                subclass = name.split(';')
                class_name.append(subclass)

    print(class_name)

    # Load Google's pre-trained Word2Vec model.
    model = gensim.models.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True)

    class_vec = []
    for ic, c in enumerate(class_name):
        this_vec = []
        print(ic)
        for i in c:
            try:
                v = model.wv[i]
                print('\t' + i)
                # print(v)
                this_vec.append(v)
            except:
                print('\t %s is not in the dic' %i )
        class_vec.append(this_vec)

    obj = dict()
    obj['word'] = class_name
    obj['vec'] = class_vec


    with open(filename, 'wb') as fp:
        pickle.dump(obj, fp, protocol=pickle.HIGHEST_PROTOCOL)


with open(filename, 'rb') as fp:
    obj = pickle.load(fp)

# print(obj)
import numpy as np

dist = np.zeros([150,150])
for ic1,c1 in enumerate(obj['vec']):
    for ic2,c2 in enumerate(obj['vec']):
        min_dis = 100000000000
        if ic1!=ic2:
            for jc1,z1 in enumerate(c1):
                for jc2, z2 in enumerate(c2):
                    tdist = np.linalg.norm(z1-z2)
                    if tdist < min_dis:
                        min_dis = tdist
        dist[ic1,ic2] = min_dis
        dist[ic2,ic1] = min_dis

mc = np.argmin(dist, axis=0)
print(mc.shape)
for i in range(mc.shape[0]):
    print(obj['word'][i])
    print("\t" + np.array2string(np.asarray(obj['word'][mc[i]])))




sorted_list = np.argsort(dist, axis=0)
print(sorted_list)

np.save(rel_filename,sorted_list)

ndist = np.load(rel_filename)
print(ndist)