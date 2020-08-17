import numpy as np
import matplotlib.pyplot as plt
import random
import sys
from joblib import Parallel, delayed
from qap_sim import quadratic_assignment_sim

from graspy.match import GraphMatch as GMP
from graspy.simulations import sbm_corr

def match_ratio(inds, n):
    return np.count_nonzero(inds == np.arange(n)) / n


n = 150
m = 100
t = 50
rhos = 0.1 * np.arange(11)[5:]
rhos = np.arange(5,10.5,0.5) *0.1
n_p = len(rhos)
ratios = np.zeros((n_p,m))
scores = np.zeros((n_p,m))

ratios_ss = np.zeros((n_p,m))
scores_ss = np.zeros((n_p,m))

n_per_block = int(n/3)
n_blocks = 3
block_members = np.array(n_blocks * [n_per_block])
block_probs = np.array([[0.2, 0.01, 0.01], [0.01, 0.1, 0.01], [0.01, 0.01, 0.2]])
directed = False
loops = False
for k, rho in enumerate(rhos):

    def run_sim(i):

        A1, A2 = sbm_corr(
            block_members, block_probs, rho, directed=directed, loops=loops
        )
        score = 0
        res_opt = None
        
        score_ss = 0
        res_opt_ss = None
        
        for j in range(t):
            seed = np.random.randint(1000)
            res = quadratic_assignment_sim(A1,A2, sim=False, maximize=True, options={'seed':seed})
            if res['score']>score:
                res_opt = res
                score = res['score']
            
            res = quadratic_assignment_sim(A1,A2, sim=True, maximize=True, options={'seed':seed})
            if res['score']>score_ss:
                res_opt_ss = res
                score_ss = res['score']
                
        ratio = match_ratio(res_opt['col_ind'], n)
        score = res_opt['score']
        
        ratio_ss = match_ratio(res_opt_ss['col_ind'], n)
        score_ss = res_opt_ss['score']
        
        return ratio, score, ratio_ss, score_ss
    
    res = Parallel(n_jobs=-1)(delayed(run_sim)(i) for i in range(m))
    
    ratios[k,:] = [item[0] for item in res]
    scores[k,:] = [item[1] for item in res]
    ratios_ss[k,:] = [item[2] for item in res]
    scores_ss[k,:] = [item[3] for item in res]
    
    
np.savetxt('ratios.csv',ratios, delimiter=',')
np.savetxt('scores.csv',scores, delimiter=',')
np.savetxt('ratios_ss.csv',ratios_ss, delimiter=',')
np.savetxt('scores_ss.csv',scores_ss, delimiter=',')