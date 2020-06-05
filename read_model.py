import numpy
import pickle

if __name__ == '__main__':
    f = open("models_v1.pkl", "rb")
    model = pickle.load(f)
    f.close()
    f = open("models_v1.txt", "w")

    for key in model.keys():
        f.write("\n++++++++++++++++\n")
        f.write(key)
        f.write("\n++++++++++++++++\nstartprob:\n")
        f.write(numpy.array_str(model[key].startprob_, precision=2, suppress_small=True))
        f.write("\n-------------\ntransmat:\n")
        f.write(numpy.array_str(model[key].transmat_, precision=2, suppress_small=True))
        f.write("\n-------------\n")

    f.close()