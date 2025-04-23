import numpy as np
from keras.models import load_model
from pkg_resources import resource_filename
from spliceai.utils import one_hot_encode
import matplotlib.pyplot as plt
import pandas as pd

paths = ('models/spliceai{}.h5'.format(x) for x in range(1, 6))
models = [load_model(resource_filename('spliceai', x)) for x in paths]

def scoreseq(seq,context=10000):
    x = one_hot_encode('N'*(context//2) + seq + 'N'*(context//2))[None, :]
    y = np.mean([models[m].predict(x) for m in range(5)], axis=0)
    acceptor_prob = y[0, :, 1]
    donor_prob = y[0, :, 2]
    nothing_prob=y[0,:,0]
    return [acceptor_prob, donor_prob, nothing_prob]
    

data=pd.read_csv("../2024_09_16_seqs/seqs/Table S13.csv",skiprows=2) 

context=10000
outf=open("spliceAI_scoresall_context=%d.csv"%context,"w")
nseqs=len(data)
print("total:%d"%nseqs)
for i in range(nseqs):
    if i%100==0:
        print(i)
        print("==============")
    event, variant, oligo, upint, var_seq, doint = data.iloc[i].values[0:6]
    seq=upint+var_seq+doint
    
    if type(seq)!=str:
        print(seq)
    else:
        acceptor, donor, nothing=scoreseq(seq)
    a_s=",".join(list(map(str,acceptor)))
    d_s=",".join(list(map(str,donor)))
    n_s=",".join(list(map(str,nothing)))

    outf.write("%s;%s;%s;%s;%s\n"%(event,variant, a_s, d_s, n_s))
outf.close()
