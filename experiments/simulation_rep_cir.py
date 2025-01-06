import time
import numpy as np
from pymatching import Matching
import sys
from os.path import abspath, dirname
sys.path.append(abspath(dirname(__file__)).strip('experiments'))
from planar import Planar_rep_cir
from codes import rep_cir
import stim
import argparse
import multiprocessing as mp
from copy import deepcopy




def count_logical_errors(detector_error_model, detection_events, observable_flips):
    # Sample the circuit.
    num_shots = detection_events.shape[0]
    matcher = Matching.from_detector_error_model(detector_error_model)

    # Run the decoder.
    predictions = matcher.decode_batch(detection_events).squeeze()
    # Count the mistakes.
    num_errors = 0
    for shot in range(num_shots):
        actual_for_shot = observable_flips[shot][0]
        predicted_for_shot = predictions[shot]
        # print(actual_for_shot, predicted_for_shot)
        if not np.array_equal(actual_for_shot, predicted_for_shot):
            num_errors += 1
    return num_errors

def mwpm(pcm, lx, error_rates, syndrome):
    weights = np.log((1-np.array(error_rates))/np.array(error_rates))
    decoder = Matching(pcm, weights=weights)
    recover = decoder.decode(syndrome)
    L = (lx@recover.T)%2
    return L

def error_generator(n, error_rate):
    error = np.random.choice([0, 1], n, p=[1-error_rate, error_rate])
    return error


def decode_simulation(d, r, error_prob, trial_num):
    fail_num = 0
    ts = 0
    circuit = stim.Circuit.generated(code_task="repetition_code:memory",
                                        distance=d,
                                        rounds=r,
                                        after_clifford_depolarization=error_prob,
                                        before_measure_flip_probability=error_prob,
                                        after_reset_flip_probability=error_prob,
                                        )

    dem = circuit.detector_error_model(decompose_errors=False, flatten_loops=True)
    rep = rep_cir(d, r)
    rep.reorder(dem)
    planar_decoder = Planar_rep_cir(rep.hx, rep.hz, rep.lx, rep.lz)
    
    num_ems = dem.num_errors
        
    er = []
    for i in range(num_ems):
        er.append(dem[i].args_copy()[0])
    
    sampler = circuit.compile_detector_sampler(seed=int(100*error_prob))
    det, obv = sampler.sample(shots=trial_num, separate_observables=True)
    
        
    stim_result = count_logical_errors(dem, det, obv)
    # print(stim_result)

    pym_result = 0
    # ex_result = 0

    ne = 0
    for trial in range(0, trial_num):

        syndrome = np.array(det[trial])
        # print(repr(syndrome))
        
        try:
            t0 = time.perf_counter()
            recover = planar_decoder.decode(syndrome, er, rep.pebz)#
            Lr = (recover @ planar_decoder.lx.T)%2
            ts += time.perf_counter() - t0
            L = obv[trial]
            if Lr!=L:
                fail_num += 1
            
            Lmw = mwpm(rep.hx, rep.lx, er, syndrome)
            if Lmw!=L:
                pym_result += 1
                
        except Exception as e:
            ne += 1
            print('Error occurred in trial {}: {}'.format(trial, e))
            print('{} errors occurred'.format(ne))
            continue
        
        

        if trial%100 == 0:
            print(trial, ts/(trial+1-ne), pym_result/(trial+1-ne), fail_num/(trial+1-ne))
        
        # print(ne)
        
    return stim_result/(trial_num-ne), pym_result/(trial_num-ne), fail_num/(trial_num-ne), ts/(trial_num-ne)
    

if __name__ == '__main__': 

    parser = argparse.ArgumentParser()

    parser.add_argument("-d", type=int, default=11)
    parser.add_argument("-r", type=int, default=11)
    parser.add_argument("-ei", type=int, default=0)
    parser.add_argument("-error_start", type=float, default=0.05)
    parser.add_argument("-error_end", type=float, default=0.1)
    parser.add_argument("-error_interval", type=int, default=10)
    parser.add_argument("-trial", type=int, default=10000)
    args = parser.parse_args()


    trial_num = args.trial
    error_rate = np.linspace(args.error_start, args.error_end, args.error_interval)[args.ei]
    
    # data = {}.fromkeys(ds)


    d = args.d
    r = args.r
    # error_prob = 0.001
   
    lo_rates = []#np.zeros(10)
    mwlo_rates = []#np.zeros((10, 10))
    threads = 10
    subt = [int(args.trial/threads)]*threads

    settings = [(
        d,
        r,
        error_rate,
        subt[i]) for i in range(threads)
    ]

    p = mp.Pool(threads)
    results = p.starmap(decode_simulation, settings)
    p.close()
    p.join()
    

    stim_results = []
    mw_results = []
    pl_results = []
    ts = []
    for i in range(threads):
        stim_results.append(results[i][0])
        mw_results.append(results[i][1])
        pl_results.append(results[i][2])
        ts.append(results[i][3])
    print(stim_results)
    print(mw_results)
    print(pl_results)
    print(ts)

