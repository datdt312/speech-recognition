import librosa
import numpy as np
import os
import math
from sklearn.cluster import KMeans
import hmmlearn.hmm
import random
from collections import defaultdict
import pickle
import warnings
warnings.filterwarnings("error")

path_to_data = "./Data_Filtered"

def get_mfcc(file_path):
    y, sr = librosa.load(file_path) # read .wav file
    hop_length = math.floor(sr*0.010) # 10ms hop
    win_length = math.floor(sr*0.025) # 25ms frame
    # mfcc is 12 x T matrix
    mfcc = librosa.feature.mfcc(
        y, sr, n_mfcc=12, n_fft=1024,
        hop_length=hop_length, win_length=win_length)
    # substract mean from mfcc --> normalize mfcc
    mfcc = mfcc - np.mean(mfcc, axis=1).reshape((-1,1)) 
    # delta feature 1st order and 2nd order
    delta1 = librosa.feature.delta(mfcc, order=1)
    delta2 = librosa.feature.delta(mfcc, order=2)
    # X is 36 x T
    X = np.concatenate([mfcc, delta1, delta2], axis=0) # O^r
    # return T x 36 (transpose of X)
    return X.T # hmmlearn use T x N matrix

def get_class_data(data_dir):
    ls = os.listdir(data_dir)
    files = [f for f in ls if f.endswith(".wav")]
    random.shuffle(files)
    mfcc = [get_mfcc(os.path.join(data_dir,f)) for f in files]
    return mfcc

def clustering(X, n_clusters=10):
    kmeans = KMeans(n_clusters=n_clusters, n_init=100, random_state=0, verbose=0)
    kmeans.fit(X)
    print("centers", kmeans.cluster_centers_.shape)
    return kmeans  

if __name__ == "__main__":

    loopi = 0
    while (loopi<12):
        try:
            loopi += 1
            print(f"_____LOOP__{loopi}___")
            class_names = ["Nha", "ThanhPho",  "Me", "YTe", "Hoc",]# "test_ThanhPho", "test_Me", "test_Nha", "test_YTe", "test_Hoc",]

            datas = {}
            dataset = {}
            for cname in class_names:
                print(f"Load {cname} dataset", end=' - ')
                datas[cname] = get_class_data(os.path.join(path_to_data, cname))
                print(len(datas[cname]))
                datas[f"test_{cname}"] = datas[cname][-20:]
                datas[cname] = datas[cname][:-20]
                #datas[f"test_{cname}"] = get_class_data(os.path.join(path_to_data, f"test_{cname}"))
                
            print("Done!!!")

            dict_components = {
                #   tʰa̤jŋ˨˩ fo˧˥ -> 5 âm vị -> 15 states
                "ThanhPho": 15,
                #  mɛ̰ʔ˨˩ -> 2 âm vị -> 6 states
                "Me": 6,
                #  i˧˧ te˧˥ -> 3 âm vị -> 9 states
                "YTe": 9,
                #  ha̰ʔwk˨ -> 3 âm vị -> 9 states
                "Hoc": 9,
                #  ɲa̤ː˨˩ -> 2 âm vị -> 6 states
                "Nha": 6,
            } 

            models = {}
            for cname in class_names:
                class_vectors = datas[cname]

                if cname[:4] != 'test':
                    n = dict_components[cname]
                    startprob = np.zeros(n)
                    startprob[0] = 1.0
                    transmat=np.diag(np.full(n,1))
                    #transmat = np.array(dict_transmat[cname])
                    
                    hmm = hmmlearn.hmm.GMMHMM(
                        n_components=n, 
                        n_mix = 4, random_state=10, n_iter=1000, verbose=True,
                        params='mctw', init_params='mct',
                        startprob_prior=startprob,
                        transmat_prior=transmat,
                    )
                
                    X = np.concatenate(datas[cname])
                    lengths = list([len(x) for x in datas[cname]])
                    print("training class", cname)
                    print(X.shape, lengths, len(lengths))
                    # FOR GMMHMM: NO NEED lengths parameter
                    hmm.fit(X)
                    models[cname] = hmm
            print("Training done")

            print("Testing")
            result = {}
            for cname in class_names:
                true_cname = f"test_{cname}"
                true_predict = 0
            #     for O in dataset[true_cname]:
                for O in datas[true_cname]:
                    score = {cname : model.score(O, [len(O)]) for cname, model in models.items()}
                    predict = max(score, key=score.get)
                    if predict == cname:
                        true_predict += 1
            #         print(true_cname, score, predict)
                result[true_cname] = f"QUANTITY: {true_predict}/{len(datas[true_cname])}\nACCURACY: {100*true_predict/len(datas[true_cname])}"
            
            for k, v in result.items():
                print(k,'\n',v,'\n')

            np.set_printoptions(precision=3, suppress=True)
            for k, v in models.items():
                print(k,v.transmat_)

            with open(f"models_v{loopi}.pkl", "wb") as file:
                pickle.dump(models, file)
        except RuntimeWarning:
            loopi -= 1
            continue