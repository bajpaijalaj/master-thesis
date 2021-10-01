#Reference: https://pymoo.org/
from pymoo.algorithms.nsga2 import NSGA2
from pymoo.factory import get_problem, get_sampling, get_crossover, get_mutation
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
import matplotlib.pyplot as plt
import autograd.numpy as anp
from pymoo.model.problem import Problem
from bitarray import bitarray
import pandas as pd
import time
import numpy as np
import ast
from pymoo.util.termination.default import MultiObjectiveDefaultTermination

class MyProblem(Problem):

    def __init__(self, const_1, features, imp, sim):

        # Lower and upper bounds
        xl = anp.array([0]*features).astype(int)
        xu = anp.array([1]*features)
        
        super().__init__(n_var=features, n_obj=2, n_constr=1, xl=xl, xu=xu)
        self.importance_scores = imp
        self.similarity_scores = sim
        self.generation = 0
        self.const_1 = const_1

    def _evaluate(self, x, out, *args, **kwargs):
        
        f1 = self.function1(x)
        f2 = self.function2(x)
        g1 = [(bitarray(i).count(1)-self.const_1) for i in x.tolist()]
        self.generation +=1
        #print(f1)
        out["F"] = anp.column_stack([f1, f2])
        #print(out["F"])
        out["G"] = anp.column_stack([g1])
    def function1(self, xx):
        output = []
        
        x = [bitarray(i) for i in xx.tolist()]
        listOfOnes = [i.search(bitarray('1')) for i in x]
        
        for y in listOfOnes:
            value = 0
            for i in y:
                value = value - self.importance_scores.loc[i]
            output.append(value)
        
        return output
    
    def function2(self, xx):
        output = []
        x = [bitarray(i) for i in xx.tolist()]
        listOfOnes = [i.search(bitarray('1')) for i in x]
        
        lengths = [len(i) for i in listOfOnes]
        for outer,length in zip(listOfOnes, lengths):
            value = 0
            for i in range(0,length-1):
                for j in range(i+1,length):
                    value = value + self.similarity_scores.loc[outer[i]][outer[j]]
            output.append(value)
        
        return output

def runNSGA2_constrained(dataset_name, constraint_zip, num_features, pop_size, ranking, imp_metric, sim_metric, fold):     
    start_time = time.time()
    output_path = "../output/"
    constraint = constraint_zip[0]
    
    imp_scores_file = open(output_path+dataset_name+"/scores/"+ranking+"_"+dataset_name+"_"+fold+"_importance_scores.txt", "r")
    text = imp_scores_file.read()
    text = text.replace("nan", "0")
    listt = []
    listt = ast.literal_eval(text)
    imp_scores_file.close()
    score_dataframe = pd.DataFrame.from_dict(listt)
    
    sim_scores_file = pd.read_csv(output_path+dataset_name+"/scores/"+ranking+"_"+dataset_name+"_"+fold+"_"+sim_metric+".csv", sep=",", header=None)
    sim_scores_file = sim_scores_file.fillna(0)
    problem = MyProblem(constraint, num_features, score_dataframe[imp_metric], sim_scores_file) 
    
    algorithm = NSGA2(pop_size, sampling=get_sampling("bin_random"), crossover=get_crossover("bin_two_point"), mutation=get_mutation("bin_bitflip"), eliminate_duplicates=True)
    termination = MultiObjectiveDefaultTermination(x_tol=1e-8,cv_tol=1e-6,f_tol=0.0025,nth_gen=5,n_last=30,n_max_gen=500,n_max_evals=100000)
    res = minimize(problem, algorithm, termination, seed=1, verbose=False)
    
    try:
        np.savetxt(output_path+dataset_name+"/fs_constrained/"+ranking+"_"+dataset_name+"_"+fold+"_"+sim_metric+"_"+imp_metric+"_"+str(constraint_zip[1])+".txt", res.X.astype(int), delimiter="", fmt='%s')#res.F
    except Exception as e:
        print(e)