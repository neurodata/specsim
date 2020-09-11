import numpy as np
import matplotlib.pyplot as plt
import random
import sys
from joblib import Parallel, delayed
from .qap_sim import quadratic_assignment_sim
import seaborn as sns
from graspy.match import GraphMatch as GMP
from graspy.simulations import sbm_corr
from .jagt import SeedlessProcrustes
from graspy.embed import AdjacencySpectralEmbed


def run_sim2(r, t, n=150, flip='median'):
    def match_ratio(inds, n):
        return np.count_nonzero(inds == np.arange(n)) / n
    def _median_sign_flips(X1, X2):
        X1_medians = np.median(X1, axis=0)
        X2_medians = np.median(X2, axis=0)
        val1 = np.sign(X1_medians).astype(int)
        X1 = np.multiply(val1.reshape(-1, 1).T, X1)
        val2 = np.sign(X2_medians).astype(int)
        X2 = np.multiply(val2.reshape(-1, 1).T, X2)
    
        return X1, X2
    #rhos = 0.1 * np.arange(11)[5:]
    m = r
    rhos = np.arange(5,10.5,0.5) *0.1
    n_p = len(rhos)
    ratios_3 = np.zeros((n_p,m))
    scores_3 = np.zeros((n_p,m))

    ratios_10 = np.zeros((n_p,m))
    scores_10 = np.zeros((n_p,m))

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
            score_10 = 0
            res_10_opt = None
            
            score_3 = 0
            res_3_opt = None

            ase3 = AdjacencySpectralEmbed(n_components=3, algorithm='truncated')
            Xhat31 = ase3.fit_transform(A1)
            Xhat32 = ase3.fit_transform(A2)

            ase10 = AdjacencySpectralEmbed(n_components=10, algorithm='truncated')
            Xhat101 = ase10.fit_transform(A1)
            Xhat102 = ase10.fit_transform(A2)

            if flip=='median':
                xhh31, xhh32 = _median_sign_flips(Xhat31, Xhat32)
                S3 = xhh31 @ xhh32.T
                xhh101, xhh102 = _median_sign_flips(Xhat101, Xhat102)
                S10 = xhh101 @ xhh102.T
            elif flip=='jagt':
                sp3 = SeedlessProcrustes().fit(Xhat31, Xhat32)
                xhh31 = Xhat31@sp3.Q_
                xhh32 = Xhat32
                S3 = xhh31 @ xhh32.T
                sp10 = SeedlessProcrustes().fit(Xhat101, Xhat102)
                xhh101 = Xhat101@sp10.Q_
                xhh102 = Xhat102
                S10 = xhh101 @ xhh102.T
            else:
                S = None
    
            for j in range(t):
                res = quadratic_assignment_sim(A1, A2, True, S3, options={'seed':seed[j]})
                if res['score']>score_3:
                    res_3_opt = res
                    score_3 = res['score']
                
                res = quadratic_assignment_sim(A1, A2, True, S10, options={'seed':seed[j]})
                if res['score']>score_10:
                    res_10_opt = res
                    score_10 = res['score']
                    
            ratio_3 = match_ratio(res_3_opt['col_ind'], n)
            score_3 = res_opt['score']
            
            ratio_10 = match_ratio(res_10_opt['col_ind'], n)
            score_10 = res_opt_ss['score']


            return ratio_3, score_3, ratio_10, score_10
        
        result = Parallel(n_jobs=-1, verbose=10)(delayed(run_sim)(seed) for seed in seeds)
        
        ratios_3[k,:] = [item[0] for item in result]
        scores_3[k,:] = [item[1] for item in result]
        ratios_10[k,:] = [item[2] for item in result]
        scores_10[k,:] = [item[3] for item in result]
        
        
    np.savetxt('ratios_3.csv',ratios_3, delimiter=',')
    np.savetxt('scores_3.csv',scores_3, delimiter=',')
    np.savetxt('ratios_10.csv',ratios_10, delimiter=',')
    np.savetxt('scores_10.csv',scores_10, delimiter=',')


    from scipy.stats import sem
    error = [2*sem(ratios_3[i,:]) for i in range(n_p)]
    average = [np.mean(ratios_3[i,:] ) for i in range(n_p)]

    error_10 = [2*sem(ratios_10[i,:]) for i in range(n_p)]
    average_10 = [np.mean(ratios_10[i,:] ) for i in range(n_p)]
    sns.set_context('paper')
    #sns.set(rc={'figure.figsize':(15,10)})
    txt =f'r={r}, t={t}'
    plt.errorbar(rhos,average_10, error_10,marker='o',capsize=3, elinewidth=1, markeredgewidth=1, label='dhat = 10')
    plt.errorbar(rhos,average, error,marker='o',capsize=3, elinewidth=1, markeredgewidth=1, label='dhat = 3', color='red')
    plt.xlabel("rho")
    plt.ylabel("avergae match ratio")
    plt.text(0.5,0.5,txt)
    plt.legend()
    plt.savefig('figure_matchratio3v101.png', dpi=150, facecolor="w", bbox_inches="tight", pad_inches=0.3)
