import numpy as np
import matplotlib.pyplot as plt

import deap
from deap import base
from deap import creator
from deap import tools
from deap import algorithms
from functools import partial
import sys, os, random
from scipy.spatial.distance import cdist


from scipy.optimize import minimize


def psi_linear_system1(parset,x, verbose=False, return_out=False):
    alpha0, c1,c2,c3_,c4,c7_,c8_,ke,kes,kis,kisr_=parset
    delta=1
    c7=c1*c7_
    c8=c2*c8_
    kisr=kis*kisr_
    kesr=kes
    c3=c3_*x
    c5=c3
    c6=c4
    #steady-state AX+B=0-->AX=-B. So negative sign in alpha0
    B=np.array([-alpha0,0,0,0,0,0]).reshape(6,1) 
    A=np.array([[-(ke+c1+c3),c2,c4,0,0,0], #P
                [c1,-(c2+c5+kes+kis),0,c6,0,0], #PS
                [c3,0,-(c4+c7+ke),c8,0,0], #PR
                [0,c5,c7,-(c8+c6+kesr+kisr),0,0], #PSR
                [ke,kes,ke,kesr,-delta,0], #E
                [0,kis,0,kisr,0,-delta]],dtype=float) #I
    #print(A)
    #print(B)
    out=np.linalg.solve(A,B) #solves AX=B
    #print(out)
    
    p,ps,pr,psr,E,I=out
    psi=100*I/(I+E)
    if verbose:
        print(out.flatten(), ", psi:", psi)
    if return_out:
        return [psi, out]
    else:
        return psi

def psi_linear_simple(parset,x, verbose=False, return_out=False):
    alpha0, c1,c2,c5_,c6,ke,kes,kis,kisr_=parset
    delta=1
    kisr=kis*kisr_
    kesr=kes
    c5=c5_*x
    
    
    B=np.array([-alpha0,0,0,0,0]).reshape(5,1)
    A=np.array([[-(ke+c1),c2,0,0,0], #P
                [c1,-(c2+c5+kes+kis),c6,0,0], #PS
                [0,c5,-(c6+kesr+kisr),0,0], #PSR
                [ke,kes,kesr,-delta,0], #E
                [0,kis,kisr,0,-delta]],dtype=float) #I
    #print(A)
    #print(B)
    out=np.linalg.solve(A,B)
    #print(out)
    
    p,ps,psr,E,I=out
    psi=100*I/(I+E)
    if verbose:
        print(out.flatten(), ", psi:", psi)
    if return_out:
        return [psi, out]
    else:
        return psi


    
def return_parsdict(groups, pars_per_group=None,pars_per_group_refine=None, fixedpars=None,
                    ratenames=None,inputnames=["GFP","CTR","LOW","MID","HIGH"], parranges=None,
                    minv=-3, maxv=3):
    """Necessary array idxs for first global optimization, then refinement of groups, then refinement of WT and endo."""
    
    """pars_per_group: list of lists. Each item is a condition (shared set of parameters).
    pars_per_group_refine: dict. 
    If parsetnames is not None, those are the parameters to be optimized. 


    """

    
    ngroups=len(pars_per_group)
    if fixedpars:
        fixedparsdict=fixedpars
        fixedparsar=np.asarray(list(fixedpars.values()),dtype=float)
    else:
        fixedparsdict=dict()
        fixedparsar=np.array([])

    getparskwargs={"groups": groups, 
                   "ngroups":ngroups, 
                   "pars_per_group":pars_per_group,
                   "pars_per_group_refine": pars_per_group_refine,
                   "ratenames": ratenames,
                   "inputnames": inputnames,
                   "fixedpars": fixedparsar,
                   "bminv": minv,
                   "bmaxv": maxv}

    inputs_plus_rates=inputnames+ratenames
    nir=len(inputs_plus_rates)
    dicts=[dict(),dict()]
    key_fix="idxs_fix"
    key_optim="idxs_optim"
    key_optim_names="names_optim"
    key_pnames="parsetnames"
    for k in range(2):
        if k==0:
            pars_per_group_=pars_per_group
            idxsdict=dicts[0]
            idxsdictname="idxsdict_global"
            
        else:
            pars_per_group_=pars_per_group_refine
            idxsdict=dicts[1]
            idxsdictname="idxsdict_refine"

        flat_list_wg = [ x for xs in pars_per_group_ for x in xs] #parameter:group
        flat_list_wog = [ x.split(":")[0] for x in flat_list_wg] #parameter
        
        
        parsetnames_fixed=[]
        #first create the list of names for each unique parameter 
        parsetnames=[]

         
        
        for x in inputs_plus_rates:
            if x in fixedparsdict.keys():
                parsetnames_fixed.append(x)
            else:
                if not x in flat_list_wog:
                    parsetnames.append(x)
                else:
                    if flat_list_wog.count(x)!=ngroups:
                        parsetnames.append(x+":c")#common input for some but not all
                    parsetnames.extend([x_ for x_ in flat_list_wg if x in x_])

        print("parsetnames_fixed", parsetnames_fixed)
        print("parsetnames", parsetnames)

        

        idxs_optim=-100*np.ones((ngroups,nir),dtype=int) 
        idxs_optim_names=np.zeros((ngroups,nir),dtype=object)
        idxs_fix=-100*np.ones((ngroups,nir),dtype=int)

        for g in range(ngroups):
            group=groups[g]
            for p,par in enumerate(inputs_plus_rates):
            
                #first try to see if that input name is common:
                name=par
                if name in parsetnames_fixed:
                    idxs_fix[g,p]=parsetnames_fixed.index(name)
                else:
                    if name in parsetnames:
                        idx=parsetnames.index(name)
                        name_save=name
                    else:
                        name_g=name+":"+group
                        
                        if name_g in parsetnames:
                            idx=parsetnames.index(name_g)
                            name_save=name_g
                        else:
                            idx=parsetnames.index(name+":c")
                            name_save=name+":c"

                    idxs_optim[g,p]=idx #index of the list of parameters that corresponds to this group-rate combination
                    idxs_optim_names[g,p]=name_save
        idxsdict[key_fix]=idxs_fix
        idxsdict[key_optim]=idxs_optim
        idxsdict[key_optim_names]=idxs_optim_names
        idxsdict[key_pnames]=parsetnames

        getparskwargs[idxsdictname]=idxsdict  
    
    
    
    
    bounds=[]
    boundsdict_=dict()
    boundsdict=dict()
    for name in getparskwargs["idxsdict_global"]["parsetnames"]:
        if ":" in name:
            name=name.split(":")[0]
        if name in parranges.keys():
            b=parranges[name]
        else:
            b=[minv,maxv]
        bounds.append(b)
        boundsdict_[name]=bounds[-1]
    getparskwargs["bounds"]=bounds
    
    #now get a 2D array with ones or zeros, whether that parameter is to be kept fixed when refining
    parsetnames_refine=[] #names of the tiled parameters upon refinement. 
    idxs_tofixrefine=np.zeros((ngroups,nir))
    for g in range(ngroups):
        group=groups[g]
        boundsdict[group]=[]
        pars_refine=[x.split(":")[0] for x in pars_per_group_refine[g]]
        for x_,x in enumerate(inputs_plus_rates):
            if not x in pars_refine:
                idxs_tofixrefine[g,x_]=1 #1 if parameter too keep fixed, 0 otherwise
            else:
                boundsdict[group].append(boundsdict_[x])

    mask_input=np.ones((ngroups,5),dtype=bool) #minigene does not have MID. 
    for g in range(ngroups):
        group=groups[g]

        if not "endo" in group and "MID" in inputnames:
            mask_input[g,inputnames.index("MID")]=False

    getparskwargs["idxsdict_refine"]["idxs_tofixrefine"]=idxs_tofixrefine

    getparskwargs["bounds"]=bounds
    getparskwargs["boundsrefine"]=boundsdict
    getparskwargs["mask_input"]=mask_input

    return getparskwargs


def out_event(parset,xvec_psi, solve_linear_system=None):
    n=len(xvec_psi)
    o1=np.zeros(n)
    for x_ in range(n):
        x=xvec_psi[x_]
        psi=solve_linear_system(parset,x)
        o1[x_]=psi
    return o1

def abserror(model,data):
    n=len(model)
    return np.sum(np.abs(model-data))/n

def mserror(model,data):
    n=len(model)
    return (1/n)*np.sum((model-data)**2)

def get_parameters_per_group(allpars_,expand=True, npars=1, ngroups=None, fixedpars=None,  parsetnames=None, idxs_optim=None, idxs_fix=None, names_optim=None,verbose=False, **kwargs):
    """
    - allpars: parameters to optimize. 
    - fixedpars: 1D array of parameters to keep fixed. E.g. global parameters to keep fixed.
    
    """

    
    
    #ngroups=kwargs["ngroups"]
    #idxs_tooptim1=kwargs["idxs_tooptim1"]
    #idxs_tofix1=kwargs["idxs_tofix1"]
    #npars=len(kwargs["inputandratenames"])
    #fixedpars=kwargs["fixedpars"]

    if expand:
        allpars=10**np.asarray(allpars_)
        allparsets=np.zeros((ngroups,npars)) 
        #this can almost surely be made more efficient with array broadcasting
        #for g in range(ngroups):
        #    #print("idxs_optim:", idxs_optim)
        #    mask=np.where(idxs_fix[g]>-99)[0]
        #    if len(mask)>0:
        #        allparsets[g,mask]=fixedpars[idxs_fix[g][mask]]
        #    mask=np.where(idxs_optim[g]>-99)[0]
        #    if len(mask)>0:
        #        allparsets[g,mask]=allpars[idxs_optim[g][mask]]
        fixed_mask = idxs_fix > -100
        optim_mask = idxs_optim > -100

        # Update allparsets using array broadcasting
        allparsets[fixed_mask] = fixedpars[idxs_fix[fixed_mask]]
        allparsets[optim_mask] = allpars[idxs_optim[optim_mask]]


        return allparsets
    else: #contract
        tiledpars=np.zeros(len(parsetnames)) #this is the names of all the parameters that are being optimized
        for i in range(len(parsetnames)): #these is the number of parameters names of the parameters that are being refine
            
            loc=np.where(names_optim==parsetnames[i])
            
            values=allpars_[loc]
            #print(parsetnames[i], values)
            z=values
            z2=np.column_stack((z,np.zeros(len(values))))
            if np.any(cdist(z2,z2)>0.001): #sanity check
                raise ValueError("Parameter values should be unique but they are not.", values)
            else:
                tiledpars[i]=values[0]
            
        return np.log10(tiledpars)





def error_and_plot(allpars_,npars=1,refined=False,data=None,additional_data=None,sysfunc=None,errorfunc=None,individual_error=False,plot=False,getparskwargs=None,plotkwargs=None,**kwargs):
    """
    pars_per_group to True if allpars_ has already each parameter per group, the first is the xvec_psi
    """

    mask_input=getparskwargs["mask_input"]
    if plot:
        nrows=plotkwargs["nrow"]
        ncols=plotkwargs["ncol"]
        titles=plotkwargs["titles"]
        fig,axes=plt.subplots(nrows,ncols,figsize=(2*ncols,2*nrows))
        if nrows>1:
            axes=axes.flatten()
        else:
            if ncols<2:
                axes=[axes]
    if refined:
        idxsdictname="idxsdict_refine"
    else:
        idxsdictname="idxsdict_global"
    allparsets=get_parameters_per_group(allpars_,npars=npars,**getparskwargs, **getparskwargs[idxsdictname])
    
    allerror=np.zeros(len(allparsets))
    ninput=5
    
    for g in range(len(allparsets)):
        xvec_psi=allparsets[g,0:ninput][mask_input[g]]
        range_n=np.arange(len(xvec_psi))
        rates=allparsets[g,ninput:]
        model_g=out_event(rates,xvec_psi,solve_linear_system=sysfunc)
        #print(model_i)
        data_g=data[g] #trend
        allerror[g]=errorfunc(model_g,data_g)
        if plot:
            ax=axes[g]
            
            if additional_data: #thinking about individual data when fitting average behaviour
                data_g_a=additional_data[g]
                for j in range(len(data_g_a)):
                    data_j=data_g_a[j]
                    ax.plot(range_n,data_j,color="lightgray")
            ax.plot(range_n,data_g,color="gray",marker="D")
            ax.plot(range_n,model_g,color="k",marker="o")
            ax.set_title("%s\nerror=%g"%(titles[g],allerror[g]))
    total_error=np.sum(allerror)
    if plot:
        fig.suptitle("total error=%g"%total_error)
        plt.tight_layout()
        plt.show()

    

    if individual_error:
        return allerror
    else:
        return total_error, #return tupple

def error_refine_singlegroup(pars, data=None,idxs_pars=None,reference_parset=None, mask_input=None, errorfunc=None, sysfunc=None):
    parset=reference_parset.copy()
    parset[idxs_pars]=10**pars
    xvec_psi=parset[0:5][mask_input]
    rates=parset[5:]
    model=out_event(rates,xvec_psi,solve_linear_system=sysfunc)
    return errorfunc(model,data)

def refine_group_pars(ar, getparskwargs=None, errorargs=None, ninit_refine=10):
    
    #refinedpars_out=refine_group_pars(ar, getparskwargs=getparskwargs,errorargs=errorargs_withplotting, ninit_refine=ninit_refine)

    """To refine group-specific parameters.
    parsetnames should be the name of the parameters in the tiled parameter array upon refining. """
    #print("refining")
    data=errorargs["data"]
    
    groups=getparskwargs["groups"]
    ngroups=getparskwargs["ngroups"]
    boundsdict=getparskwargs["boundsrefine"]
    mask_input=getparskwargs["mask_input"]
    fixedpars=getparskwargs["fixedpars"]
    npars=len(getparskwargs["inputnames"])+len(getparskwargs["ratenames"])
    idxs_tofixrefine=getparskwargs["idxsdict_refine"]["idxs_tofixrefine"]

    parset_initial=np.asarray(ar)
    
    allparsets=get_parameters_per_group(parset_initial,expand=True, npars=npars, **getparskwargs,**getparskwargs["idxsdict_global"],verbose=False)

    
    allparsets_optim=[]


    error=0
    for g in range(ngroups):
        group=groups[g]
        #print("group",g)
        reference_parset=allparsets[g].copy()
        idxs_pars_group=np.where(1-idxs_tofixrefine[g])[0] #idxs_tofixrefine is 1 if parameter to be fixed. If I do 1-, that turns 0, and if not to be fixed, turns 1
        tooptimpars=reference_parset[idxs_pars_group]
        bounds_group=boundsdict[group]
        #print(idxs_rates_group)
        x0=np.log10(tooptimpars)
        minfunc=partial(error_refine_singlegroup,data=data[g],idxs_pars=idxs_pars_group, mask_input=mask_input[g], reference_parset=reference_parset, errorfunc=errorargs["errorfunc"], sysfunc=errorargs["sysfunc"])

        initial_conditions=[x0]
        
        for i in range(ninit_refine):
            np.random.seed(i)
            x0_=[]
            for b in bounds_group:
                x0_.append(np.random.uniform(b[0],b[1]))
            initial_conditions.append(x0_)

        optimscores=[]
        optimpars=[]
        for x0 in initial_conditions:
            out=minimize(minfunc,x0,bounds=bounds_group)
            optimscores.append(out.fun)
            optimpars.append(out.x)
        bestidx=np.argsort(optimscores)[0]
        bestpars=optimpars[bestidx]
        error+=optimscores[bestidx]
        #print("bestidx", bestidx)
        
        optim_parset=reference_parset.copy()
        optim_parset[idxs_pars_group]=10**bestpars
        allparsets_optim.append(optim_parset)

    refined_pars_tiled=get_parameters_per_group(np.array(allparsets_optim),expand=False, **getparskwargs, **getparskwargs["idxsdict_refine"])

    
    return [refined_pars_tiled, error]

def run_genetic(bounds=None,POPULATION_SIZE=10,MAX_GENERATIONS=10, 
              P_CROSSOVER=0.1, P_MUTATION=0.1,MUTSIGMA=1,INDPB=0.25,HALL_OF_FAME_SIZE = 10,fitnessfunc=None, seeds=None, 
              plot_fitness_evo=True, plotintermediates=True, plotbest=True, errorargs_withplotting=None,
              getparskwargs=None, ninit_refine=10):
    bounds=getparskwargs["bounds"]
    def checkBounds(a):
        def decorator(func):
            def wrapper(*args, **kargs):
                offspring = func(*args, **kargs)
                for child in offspring:
                    for i in range(len(child)):
                        min_,max_=bounds[i]
                        if child[i] > max_:
                            child[i] = max_
                        elif child[i] < min_:
                            child[i] = min_
                return offspring
            return wrapper
        return decorator



    
    toolbox = base.Toolbox() 
    toolbox.register("evaluate", fitnessfunc) # fitness calculation (error, to be minimized)
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,)) # define a single objective, minimizing error


    #now register the appropriate random number generators for each unique bound limit
    unique_bounds_str=[]
    for bound in bounds:
        unique_bounds_str.append("%s;%s"%(bound[0],bound[1]))
    unique_bounds_str=np.unique(unique_bounds_str)
    print("unique_bounds_str", unique_bounds_str)
    for i in range(len(unique_bounds_str)):
        minv,maxv=list(map(float,unique_bounds_str[i].split(";")))
        toolbox.register("randuni_%d"%i, np.random.uniform, minv, maxv) # create an operator that returns a random number between min and max
    

    creator.create("Individual", list, fitness=creator.FitnessMin)
    list_rand=[]
    for p in range(len(bounds)):
        bound=bounds[p]
        bounds_str="%s;%s"%(bound[0],bound[1])
        idx=np.where(unique_bounds_str==bounds_str)[0][0]
        name_rand="toolbox.randuni_%d"%idx
        list_rand.append(eval(name_rand))

    toolbox.register("individualCreator", tools.initCycle, creator.Individual,tuple(list_rand),n=1)
    #toolbox.register("individualCreator", tools.initRepeat, creator.Individual, toolbox.randuni_1, NPARS)
    toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)# create the population operator to generate a list of individuals:

    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.decorate("mate", checkBounds(1))

    #if genetic:
    toolbox.register("select", tools.selTournament, tournsize=3)# Tournament selection with tournament size of 3:
        
    #mutation
    #make the mutation decrease over time, both reducing sigma and the mutation probability
    #from here: https://stackoverflow.com/questions/58990269/deap-make-mutation-probability-depend-on-generation-number
    #toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=CallbackProxy(lambda: max(0.5,MUTSIGMA*(1-(0.1)*(N_GEN//N_GEN_DECAY)))),indpb=CallbackProxy(lambda: max(1/NPARS,INDPB*(1-(0.1)*(N_GEN//N_GEN_DECAY)))))
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=MUTSIGMA,indpb=INDPB)
    toolbox.decorate("mutate", checkBounds(1))

    tops=[]

    for seed in seeds: #check how different seeds produce equally good results but with different parameter setsalgorithm
        print("seed", seed)
        sys.stdout.flush()
        random.seed(seed)
        np.random.seed(seed)
        population = toolbox.populationCreator(n=POPULATION_SIZE)

        # prepare the statistics object:
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("min", np.min)
        stats.register("avg", np.mean)

        # perform the Genetic Algorithm flow:
        hof = tools.HallOfFame(HALL_OF_FAME_SIZE)
        
        
        population, logbook = algorithms.eaSimple(population, toolbox, cxpb=P_CROSSOVER, mutpb=P_MUTATION, ngen=MAX_GENERATIONS,
                                   stats=stats, verbose=False, halloffame=hof)
            
        
        # Evolutionary Algorithm is done - extract statistics:
        minFitnessValues, meanFitnessValues = logbook.select("min", "avg")
        #print("Initial fitness values:", minFitnessValues[0:10])
        #print("Final fitness values:", minFitnessValues[-10:])

        if plot_fitness_evo:
            # plot statistics:
            fig,ax=plt.subplots(1,1,figsize=(3,2))

            ax.plot(minFitnessValues, color='red')
            ax.plot(meanFitnessValues, color='green')
            ax.set_xlabel('Generation')
            ax.set_ylabel('Min / Average Fitness')
            ax.set_title('Min and Average Fitness over Generations')
            plt.tight_layout()
            plt.show()
        
        #look at bests individuals and further refine -- we can just look at the hall of fame
        #fitness_list=[float(ind.fitness.values[0]) for ind in population]
        #argsort=np.argsort(fitness_list)
        #top5=argsort[0:5]
        parsetnames=getparskwargs["idxsdict_global"]["parsetnames"]
        for ind in hof.items:
            ar=np.asarray(ind)
            fitness=fitnessfunc(ar)[0] #fitnessfunc returns a tupple
            tops.append([fitness,parsetnames,ar,seed, "hof"])
            if False: #plotintermediates:
                print("HOF ITEM")
                print(",".join(list(map(str,parsetnames))))
                print(fitness)
                print(",".join(list(map(str,ar))))
                error_and_plot(ar,getparskwargs=getparskwargs,**errorargs_withplotting)
            refinedpars_out=refine_group_pars(ar, getparskwargs=getparskwargs,errorargs=errorargs_withplotting, ninit_refine=ninit_refine)
            fitness=refinedpars_out[1]
            if plotintermediates:
                print("after refinement")
                print("fitness", fitness)
                print(",".join(list(map(str,refinedpars_out[0]))))
                #plot parameter values
                fig1,ax=plt.subplots(1,1,figsize=(12,3),sharex=True)
                ax.scatter(range(len(refinedpars_out[0])),refinedpars_out[0])
                ax.set_xticks(range(len(refinedpars_out[0])))
                ax.set_xticklabels(getparskwargs["idxsdict_refine"]["parsetnames"],rotation=90)
                ax.grid("on")
                plt.show()
                error_and_plot(refinedpars_out[0],refined=True,getparskwargs=getparskwargs,**errorargs_withplotting)
            
            tops.append([fitness,getparskwargs["idxsdict_refine"]["parsetnames"],refinedpars_out[0],seed,"refined"])
    #print("tops")
    #for x in tops:
    #    print(x)
    argsort=np.argsort([x[0] for x in tops])
    best=tops[argsort[0]]
    if plotbest:
        print("best parameter set")
        print("fitness", best[0], "seed", best[2])
        if best[4]=="refined":
            refined=True
        else:
            refined=False
        ar=best[2]
        print(",".join(list(map(str,ar))))
        error_and_plot(ar,refined=refined,**errorargs_withplotting, getparskwargs=getparskwargs)

    return best
            
        
#FIT_INDIVIDUAL
#FIT_ENDO
    




