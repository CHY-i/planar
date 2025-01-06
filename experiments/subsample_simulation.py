import time
import numpy as np
from pymatching import Matching
import sys
from os.path import abspath, dirname
sys.path.append(abspath(dirname(__file__)).strip('experiments'))
from planar import Planar_rep_cir, subsamples
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

def loading_data(label, basis=None):
    path = abspath(dirname(__file__))+'/sample'+label
    print(path)
    None
# loading_data('00')




def subsample_decode_simulation(ds, d, r, error_prob, trial_num):
    pym_result = 0
    fail_num = 0
    ts = 0
    circuit0 = stim.Circuit.generated(code_task="repetition_code:memory",
                                        distance=d,
                                        rounds=r,
                                        after_clifford_depolarization=error_prob,
                                        before_measure_flip_probability=error_prob,
                                        after_reset_flip_probability=error_prob,
                                        )

    dem = circuit0.detector_error_model(decompose_errors=False, flatten_loops=True)
    sampler = circuit0.compile_sampler(seed=int(100*error_prob)) #compile_detector_sampler(seed=int(100*error_prob))
    samples = sampler.sample(shots=trial_num)
    det, obv = circuit0.compile_m2d_converter().convert(measurements=samples, separate_observables=True)



    if ds<d:
        subsample = subsamples(ds, d, r, dem)

        circuit1 = stim.Circuit.generated(code_task="repetition_code:memory", distance=ds, rounds=r,
                                          after_clifford_depolarization=error_prob,
                                          before_measure_flip_probability=error_prob,
                                          after_reset_flip_probability=error_prob,)
        dems = circuit1.detector_error_model(decompose_errors=False, flatten_loops=True)
        rep = rep_cir(ds, r)
        rep.reorder(dems)
        planar_decoder = Planar_rep_cir(rep.hx, rep.hz, rep.lx, rep.lz)
        
        # er1 = []
        # for i in range(dems.num_errors):
        #     er1.append(dems[i].args_copy()[0])

        
        for i in range(len(subsample)):
            detectors = subsample[i]['det']
            er = subsample[i]['er']
            # print(subsample[i]['em'])
            # print(np.array(er)-np.array(er1))
            # print(er)
            dets = np.array(det)[:, detectors]
            obvs = samples[:, -d+ds-1+i]
            
            # print(dets)

            
            for trial in range(trial_num):
                syndrome = dets[trial]
                L = obvs[trial]

                t0 = time.perf_counter()
                recover = planar_decoder.decode(syndrome, er, rep.pebz)#
                Lr = (recover @ planar_decoder.lx.T)%2
                ts += time.perf_counter() - t0
                if Lr!=L:
                    fail_num += 1
                
                Lmw = mwpm(rep.hx, rep.lx, er, syndrome)
                if Lmw!=L:
                    pym_result += 1
            print(fail_num, pym_result)
        fail_num = fail_num/(d-ds+1)
        pym_result = pym_result/(d-ds+1)
        ts = ts/(d-ds+1)
    else:
        rep = rep_cir(d, r)
        rep.reorder(dem)
        planar_decoder = Planar_rep_cir(rep.hx, rep.hz, rep.lx, rep.lz)
        num_ems = dem.num_errors
            
        er = []
        for i in range(num_ems):
            er.append(dem[i].args_copy()[0])


       
        # ex_result = 0
        for trial in range(trial_num):
            # print(trial)

            syndrome = np.array(det[trial])
            L = obv[trial][0]

            t0 = time.perf_counter()
            recover = planar_decoder.decode(syndrome, er, rep.pebz)#
            Lr = (recover @ planar_decoder.lx.T)%2
            ts += time.perf_counter() - t0
            if Lr!=L:
                fail_num += 1
            
            Lmw = mwpm(rep.hx, rep.lx, er, syndrome)
            if Lmw!=L:
                pym_result += 1

            
    return pym_result/trial_num, fail_num/trial_num, ts
    # return None

if __name__ == '__main__': 

    parser = argparse.ArgumentParser()

    parser.add_argument("-d", type=int, default=5)
    parser.add_argument("-r", type=int, default=3)
    parser.add_argument("-error_start", type=float, default=0.05)
    parser.add_argument("-error_end", type=float, default=0.1)
    parser.add_argument("-error_interval", type=int, default=10)
    parser.add_argument("-trial", type=int, default=10000)
    args = parser.parse_args()


    trial_num = args.trial
    error_rates = np.linspace(args.error_start, args.error_end, args.error_interval)



    d = args.d
    r = args.r
    
    dlist = [3]
   
    results = {}.fromkeys(dlist)#np.zeros(10)
    # mwlo_rates = []#np.zeros((10, 10))
    for ds in dlist:
        results[ds] = {'mw':[], 'pl':[], 'tpl':[]}
        print('distance:', ds)
        for i in range(len(error_rates)): #len(error_rates)
            mw_result, pl_result, ts =subsample_decode_simulation(ds, d, r, error_prob=error_rates[i], trial_num=trial_num)
            results[ds]['mw'].append(mw_result)
            results[ds]['pl'].append(pl_result)
            results[ds]['tpl'].append(ts/trial_num)
            # print(mw_result)
            # print(pl_result)
            # print(ts/trial_num)
    print(results)



# np.save(abspath(dirname(__file__))+'/rep_cir_2d.npy', data)
# print(data)