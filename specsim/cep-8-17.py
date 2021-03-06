import numpy as np
import matplotlib.pyplot as plt
import random
import sys
from joblib import Parallel, delayed
from qap_sim import quadratic_assignment_sim
import seaborn as sns
from graspy.match import GraphMatch as GMP
from graspy.simulations import sbm_corr

def match_ratio(inds, n):
    return np.count_nonzero(inds == np.arange(n)) / n


n = 150
m = 100
t = 60
#rhos = 0.1 * np.arange(11)[5:]
rhos = np.arange(5,10.5,0.5) *0.1
n_p = len(rhos)
ratios = np.zeros((n_p,m))
scores = np.zeros((n_p,m))

ratios_ss = np.zeros((n_p,m))
scores_ss = np.zeros((n_p,m))

ratios_opt = np.zeros((n_p,m))
scores_opt = np.zeros((n_p,m))

ratios_opt_ss = np.zeros((n_p,m))
scores_opt_ss = np.zeros((n_p,m))

n_per_block = int(n/3)
n_blocks = 3
block_members = np.array(n_blocks * [n_per_block])
block_probs = np.array([[0.2, 0.01, 0.01], [0.01, 0.1, 0.01], [0.01, 0.01, 0.2]])
directed = False
loops = False
for k, rho in enumerate(rhos):
    np.random.seed(8888)
    seeds = [np.random.randint(1e8, size=t) for i in range(m)]
    def run_sim(seed):

        A1, A2 = sbm_corr(
            block_members, block_probs, rho, directed=directed, loops=loops
        )
        score = 0
        res_opt = None
        
        score_ss = 0
        res_opt_ss = None
        
        for j in range(t):
            res = quadratic_assignment_sim(A1,A2, sim=False, maximize=True, options={'seed':seed[j]})
            if res['score']>score:
                res_opt = res
                score = res['score']
            
            res = quadratic_assignment_sim(A1,A2, sim=True, maximize=True, options={'seed':seed[j]})
            if res['score']>score_ss:
                res_opt_ss = res
                score_ss = res['score']
                
        ratio = match_ratio(res_opt['col_ind'], n)
        score = res_opt['score']
        
        ratio_ss = match_ratio(res_opt_ss['col_ind'], n)
        score_ss = res_opt_ss['score']

        res = quadratic_assignment_sim(A1,A2, sim=False, maximize=True, options={'shuffle_input':False})
        ratio_opt = match_ratio(res['col_ind'], n)
        score_opt = res['score']

        res = quadratic_assignment_sim(A1,A2, sim=True, maximize=True, options={'shuffle_input':False})
        ratio_opt_ss = match_ratio(res['col_ind'], n)
        score_opt_ss = res['score']


        return ratio, score, ratio_ss, score_ss, ratio_opt, score_opt, ratio_opt_ss, score_opt_ss
    
    result = Parallel(n_jobs=-1, verbose=10)(delayed(run_sim)(seed) for seed in seeds)
    
    ratios[k,:] = [item[0] for item in result]
    scores[k,:] = [item[1] for item in result]
    ratios_ss[k,:] = [item[2] for item in result]
    scores_ss[k,:] = [item[3] for item in result]
    ratios_opt[k,:] = [item[4] for item in result]
    scores_opt[k,:] = [item[5] for item in result]
    ratios_opt_ss[k,:] = [item[6] for item in result]
    scores_opt_ss[k,:] = [item[7] for item in result]
    
    
np.savetxt('ratios.csv',ratios, delimiter=',')
np.savetxt('scores.csv',scores, delimiter=',')
np.savetxt('ratios_ss.csv',ratios_ss, delimiter=',')
np.savetxt('scores_ss.csv',scores_ss, delimiter=',')
np.savetxt('ratios_opt.csv',ratios_opt, delimiter=',')
np.savetxt('scores_opt.csv',scores_opt, delimiter=',')
np.savetxt('ratios_opt_ss.csv',ratios_opt_ss, delimiter=',')
np.savetxt('scores_opt_ss.csv',scores_opt_ss, delimiter=',')

from scipy.stats import sem
error = [2*sem(ratios[i,:]) for i in range(n_p)]
average = [np.mean(ratios[i,:] ) for i in range(n_p)]

error_ss = [2*sem(ratios_ss[i,:]) for i in range(n_p)]
average_ss = [np.mean(ratios_ss[i,:] ) for i in range(n_p)]
sns.set_context('paper')
#sns.set(rc={'figure.figsize':(15,10)})
plt.errorbar(rhos,average_ss, error_ss,marker='o',capsize=3, elinewidth=1, markeredgewidth=1, label='GM+SS')
plt.errorbar(rhos,average, error,marker='o',capsize=3, elinewidth=1, markeredgewidth=1, label='GM', color='red')
plt.xlabel("rho")
plt.ylabel("avergae match ratio")
plt.legend()
plt.savefig('r_100_t_50.png', dpi=150, facecolor="w", bbox_inches="tight", pad_inches=0.3)
