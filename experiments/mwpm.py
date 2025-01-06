import time
import numpy as np
from pymatching import Matching
import sys
from os.path import abspath, dirname, exists
sys.path.append(abspath(dirname(__file__)).strip('experiments'))
from planar import Planar_rep_cir, Data, d2m
from codes import rep_cir
import stim
import argparse
import multiprocessing as mp
from copy import deepcopy


def count_logical_errors(pcm, lx, error_rates,detection_events, observable_flips):
    # Sample the circuit.
    num_shots = detection_events.shape[0]
    weights = np.log((1-np.array(error_rates))/np.array(error_rates))
    matcher = Matching(pcm, weights=weights)
    operator = matcher.decode_batch(detection_events).squeeze()
    predictions = (operator @ lx.T)%2
    # Count the mistakes.
    num_errors = 0
  
    for shot in range(num_shots):
        actual_for_shot = observable_flips[shot]
        predicted_for_shot = bool(predictions[shot])

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


def subsamples(ds, d, r, dem):
    n = dem.num_errors
    ns = d-ds+1
    subsample = {}.fromkeys(range(ns))
    
    for i in range(ns):
        subsample[i] = ({'det':[], 'em':[], 'er':[], 'merge_e':[]})
        for k in range(r+1):
            for j in range(ds-1):
                subsample[i]['det'].append(i+j+k*(d-1))

        detectors = subsample[i]['det']
        for j, e in enumerate(dem[:n]):
            if j not in subsample[i]['em']:
                Dec = e.targets_copy()
                if len(Dec) == 1 and str(Dec[0]).startswith('D') :
                    if int(str(Dec[0])[1:]) in detectors:
                        subsample[i]['em'].append(j)
                        subsample[i]['er'].append(e.args_copy()[0])
                else:
                    D0 = int(str(Dec[0])[1:])
                    D1 = int(str(Dec[1])[1:])
                    if str(Dec[0]).startswith('D') and str(Dec[1]).startswith('L') and D0 in detectors:
                        # print(str(Dec[0]), str(Dec[1]))
                        # print(j, 'A')
                        subsample[i]['em'].append(j)
                        subsample[i]['er'].append(e.args_copy()[0])
                    elif str(Dec[0]).startswith('D') and str(Dec[1]).startswith('D'):
                        if D0 in detectors and D1 in detectors:
                            # print(str(Dec[0]), str(Dec[1]))
                            # print(j, 'B')
                            subsample[i]['em'].append(j)
                            subsample[i]['er'].append(e.args_copy()[0])
                        elif  D0 not in detectors and D1 in detectors:
                            if D1-D0==1:
                                # print(str(Dec[0]), str(Dec[1]))
                                # print(j, 'C')
                                subsample[i]['em'].append(j)
                                
                                if D1 == detectors[0]:
                                    subsample[i]['er'].append(e.args_copy()[0])
                                else:
                                    subsample[i]['er'].append(e.args_copy()[0]*(1-dem[j-3*(d-1)+2].args_copy()[0]) + (1-e.args_copy()[0])*dem[j-3*(d-1)+2].args_copy()[0])
                                    subsample[i]['merge_e'].append([(j, j-3*(d-1)+2), (D0, D1), (int(str(dem[j-3*(d-1)+2].targets_copy()[0])[1:]), int(str(dem[j-3*(d-1)+2].targets_copy()[1])[1:]))])         
                        elif  D0 in detectors and D1 not in detectors:
                            if D1-D0==1:
                                # print(str(Dec[0]), str(Dec[1]))
                                # print(j, 'E')
                                subsample[i]['em'].append(j)
                                if D0 == detectors[-1]:
                                    subsample[i]['er'].append(e.args_copy()[0])
                                else:
                                    # print(j, j+2)
                                    subsample[i]['em'].insert(-1, j+1)
                                    subsample[i]['er'].append(dem[j+1].args_copy()[0])
                                    subsample[i]['er'].append(e.args_copy()[0]*(1-dem[j+2].args_copy()[0]) + (1-e.args_copy()[0])*dem[j+2].args_copy()[0])                     
                                    subsample[i]['merge_e'].append([(j, j+2), (D0, D1), (int(str(dem[j+2].targets_copy()[0])[1:]), int(str(dem[j+2].targets_copy()[1])[1:]))])
                    else:
                        None

    return subsample

def loading_data(label, basis, all_data = True):
    path_dem = abspath(dirname(__file__)).strip('experiments')+'/experiment_data/google/'+basis+'/sample_'+label+'/decoding_results/MWPM_decoder_with_RL_optimized_prior'+'/error_model.dem'
    dem = stim.DetectorErrorModel.from_file(path_dem)# path_mw = abspath(dirname(__file__))+'/sample'+label+'/decoding_results/MWPM_decoder_with_RL_optimized_prior'+'/obs_flips_actual.b8'
    # path_mwpre = abspath(dirname(__file__)).strip('experiments')+'/experiment_data/google/'+basis+'/sample_'+label+'/decoding_results/MWPM_decoder_with_RL_optimized_prior'+'/obs_flips_predicted.b8'
    # D1 = Data(d=29, r=1001, file_path_detection=None, file_path_measurement=None, file_path_logical_flip=path_mwpre)
    # mwpre = D1.logical_flip()
    if all_data == True:
        path_s = abspath(dirname(__file__)).strip('experiments')+'/experiment_data/google/'+basis+'/sample_'+label+'/detection_events.b8'
        path_l = abspath(dirname(__file__)).strip('experiments')+'/experiment_data/google/'+basis+'/sample_'+label+'/obs_flips_actual.b8'
        # path_m = abspath(dirname(__file__)).strip('experiments')+'/experiment_data/google/'+basis+'/sample_'+label+'/measurements.b8'
        D = Data(d=29, r=1001, file_path_detection=path_s, file_path_measurement=None, file_path_logical_flip=path_l)
        det = D.detection()
        obv = D.logical_flip()
        # meas = D.measurement()
        dem = stim.DetectorErrorModel.from_file(path_dem)
        return dem, det, obv
    else:
        return dem

    # return None


def mpsubsample_matching(trial_num, subsample, det, obvs, pcm, lx):
    detectors = subsample['det']
    er = subsample['er']
    dets = np.array(det)[:, detectors]
    pym_result = count_logical_errors(pcm, lx, er, dets, obvs)
    return {'mw_results':pym_result/trial_num}

if __name__ == '__main__': 

    parser = argparse.ArgumentParser()

    parser.add_argument("-ds", type=int, default=3)
    parser.add_argument("-d", type=int, default=29)
    parser.add_argument("-r", type=int, default=1001)
    parser.add_argument("-basis", type=str, default='X')
    parser.add_argument("-label", type=str, default='00')
    parser.add_argument("-mp_task", type=str, default='subsample')
    parser.add_argument("-trial", type=int, default=100000)
    args = parser.parse_args()

    trial_num = args.trial
    d = args.d
    r = args.r
    ds = args.ds
    
    dem, det, obv = loading_data(args.label, args.basis, True)

    mea = d2m(obv, det, d, r)
    dmea = np.array(mea)[:, -d:]

    

    if args.mp_task == 'subsample':
        circuit1 = stim.Circuit.generated(code_task="repetition_code:memory", distance=ds, rounds=r,
                                        after_clifford_depolarization=0.1,
                                        before_measure_flip_probability=0.1,
                                        after_reset_flip_probability=0.1,)
        subdem = circuit1.detector_error_model(decompose_errors=False, flatten_loops=True)
        rep = rep_cir(ds, r)
        rep.reorder(subdem)
        planar_decoder = Planar_rep_cir(rep.hx, rep.hz, rep.lx, rep.lz)

        subsamples = subsamples(ds, d, r, dem)
        threads = d-ds+1

        settings = [(
        trial_num,
        subsamples[i],
        det[:trial_num],
        dmea[:, ds-1+i],
        rep.hx,
        rep.lx,      
        ) for i in range(threads)]

        p = mp.Pool(threads)
        results = p.starmap(mpsubsample_matching, settings)
        p.close()
        mw = []
        pl = []
        for i in range(threads):
            print('mw_results', results[i]['mw_results'])
            mw.append(results[i]['mw_results'])
        print(np.array(mw).mean())
    else:
        None



# np.save(abspath(dirname(__file__))+'/rep_cir_2d.npy', data)
# print(data)