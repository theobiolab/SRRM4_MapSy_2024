import deap
from deap import base
from deap import creator
from deap import tools
from deap import algorithms
from functools import partial
import random, sys

import numpy as np
from scipy.optimize import minimize
import copy
import json
from collections import OrderedDict
import os

#sys.path.append("/Users/rosamartinezcorral/OneDrive - CRG - Centre de Regulacio Genomica/papers/github_repos/SRRM4_MapSy_2024")

sys.path.append("/users/romartinez/romartinez/repos/SRRM4_MapSy_2024")
import auxfuncs_fitting
sysfunc=auxfuncs_fitting.psi_linear_system_noalpha_lessparsv2



ratenames="c2,c4,c7,ke,kis".split(",") #parameters of the function to calculate psi
inputnames_mini="GFP,CTR,LOW,HIGH".split(",") #minigene
inputnames_endo="GFP,CTR,LOW,HIGH".split(",") #endogenous



dataf=sys.argv[1]
ratesdif_str=sys.argv[2] #string of ; separated rates that differ among categories
outfolder=sys.argv[3]
jid=int(sys.argv[4])
ratesdif=ratesdif_str.split(";")
print("ratesdif", ratesdif)
outfolder=outfolder.replace(";","")
outf=os.path.join(outfolder,"%s_out_%d.txt"%(dataf.strip(".json"),jid))

outdata=dict()

outdata["ratesdif"]=ratesdif_str


with open(dataf) as f:
    d = json.load(f)

allgroups_subset=[]
mediandata_subset=[]
for k,v in d.items():
    if not "U1cons_a" in k:
        allgroups_subset.append(k)
        if "endo" in k:
            v=list(np.asarray(v)[[0,1,2,4]])
        mediandata_subset.append(v)

groups=allgroups_subset
individual_subdfs=None
mediandata=mediandata_subset

ngroups=len(groups)

minv=-6
maxv=6

parranges={'c7':[0,maxv],'c8':[minv,0],"LOW":[0,3],"HIGH":[0,3]} #log10 scale #minigene does not have MID, careful because in the paper the nomenclature is different. CTR here is low in the paper, and LOW here is mid in the paper, HIGH here is HIGH in the paper 

fixedpars=OrderedDict({"GFP":0,"CTR":1})  #natural scale. 

bestpars_perratesdif=dict()


pars_per_group=[] #can be different
for group in groups:
    rates_=[]
    for rate in ratesdif:
        rates_.append("%s:%s"%(rate,group))
    pars_per_group.append(rates_)
print(pars_per_group)
outdata["pars_per_group"]=pars_per_group

pars_per_group_refine=pars_per_group


getparskwargs=auxfuncs_fitting.return_parsdict(groups,inputnames=inputnames_endo,pars_per_group=pars_per_group, pars_per_group_refine=pars_per_group_refine,ratenames=ratenames, minv=minv, maxv=maxv, parranges=parranges,fixedpars=fixedpars)
parsetnames=getparskwargs["idxsdict_global"]["parsetnames"]
outdata["parsetnames"]=parsetnames
outdata["bounds"]=getparskwargs["bounds"]

getparskwargs["rates_condition_refine"]=pars_per_group_refine

NPARS=len(parsetnames) 
#print(NPARS, "parameters")
errorargs={"data":mediandata,"additional_data":None,
           "npars":len(getparskwargs["inputnames"])+len(getparskwargs["ratenames"]),
      "sysfunc":sysfunc,
      "errorfunc":auxfuncs_fitting.mserror,
      "individual_error":False,
      "plot":False,
      "plotkwargs":None,
      "penaltyinput":10000}


errorargs_withplotting=errorargs.copy()
errorargs_withplotting["plotkwargs"]={"nrow":0,"ncol":0,"titles":None}
errorargs_withplotting["plot"]=True
errorargs_withplotting["additional_data"]=None #[x.values for x in individual_subdfs]

fitnessfunc=partial(auxfuncs_fitting.error_and_plot_increasinginput,getparskwargs=getparskwargs,**errorargs)


POPULATION_SIZE = 400
MAX_GENERATIONS = 300

P_CROSSOVER = 0.25  # probability for crossover
ETACXBIN = 0.25
P_MUTATION = 0.5   # probability for mutating an individ
MUTSIGMA=0.25 #sigma
INDPB=0.5


pars_genetic={"POPULATION_SIZE":POPULATION_SIZE, "MAX_GENERATIONS": MAX_GENERATIONS, 
              "P_CROSSOVER": P_CROSSOVER,"P_MUTATION": P_MUTATION, "MUTSIGMA": MUTSIGMA, "INDPB": INDPB,
              "HALL_OF_FAME_SIZE": 3,
             "fitnessfunc": fitnessfunc,
             "seeds":[jid], 
              "plot_fitness_evo":False, 
              "plotintermediates":False, 
              "plotbest":False, 
              "errorargs_withplotting":errorargs_withplotting, 
              "getparskwargs":getparskwargs,
              "cxbin":True,
              "etacxbin":ETACXBIN}

best=auxfuncs_fitting.run_genetic(**pars_genetic)
bestpars_perratesdif[ratesdif_str]=best
fitness,parsetnames,pars,seed,refinedbool=best

outdata["fitness"]=fitness
outdata["refined"]=refinedbool
outdata["parsetnames"]=parsetnames
outdata["pars"]=list(map(str,pars))


with open(outf, 'w') as fout:
    json.dump(outdata, fout)


