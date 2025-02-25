import time
import numpy as np
from pymatching import Matching
from qecsim.models.planar import PlanarCode
from scipy.linalg import block_diag
import sys
from os.path import abspath, dirname
sys.path.append(abspath(dirname(__file__)).strip('experiments'))
print(abspath(dirname(__file__)).strip('experiments'))
from planar import Planar, generate_error, cal_syndrome, decoding_success
from codes import gen_surface_code, gen_rotated_surface_code
import argparse
import multiprocessing as mp


class pymatching_decoder_warpper:
    def __init__(self, hx, hz) -> None:
        self.hx = hx
        self.hz = hz
        self.n = self.hx.shape[1]
        assert self.n == self.hz.shape[1]
        self.check_x_length = self.hx.shape[0]

    def _set_up_decoder(self, error_prob):
        # decoder for Z errors
        pz = error_prob[2] + error_prob[3]
        self.matching_z = Matching.from_check_matrix(
            self.hx,
            weights=np.log((1-pz)/pz)
        )

        # decoder for X-errors
        px = error_prob[1] + error_prob[3]
        self.matching_x = Matching.from_check_matrix(
            self.hz,
            weights=np.log((1-px)/px)
        )

    def decode(self, syndrome, s):
        syndrome_x, syndrome_z = syndrome[:self.check_x_length], syndrome[self.check_x_length:]
        recovery_z = self.matching_z.decode(syndrome_z)
        recovery_x = self.matching_x.decode(syndrome_x)

        return np.hstack([recovery_x, recovery_z])


def decode_subroutine(decoders, error_type, error_rate, trial_num, stabilizers, lx, lz):
    if error_type == 'depolarizing':
        error_prob = np.array([1-error_rate, error_rate/3, error_rate/3, error_rate/3])
    elif error_type == 'z':
        error_prob = np.array([1-error_rate, 0.0, error_rate, 0.0])
    elif error_type == 'x':
        error_prob = np.array([1-error_rate, error_rate, 0.0, 0.0])
    elif error_type == 'uncorrelated':
        error_prob = np.array([(1-error_rate)**2, error_rate(1-error_rate), error_rate(error_rate), error_rate**2])
    else:
        raise ValueError('Unkown error type.')
    rng = np.random.default_rng(int(1000*error_rate))
    for decoder in decoders:
        if type(decoder) == pymatching_decoder_warpper:
            decoder._set_up_decoder(error_prob)
    fail_num = [0] * len(decoders)
    ts = [0] * len(decoders)

    for trial in range(trial_num):
        error = generate_error(n, error_prob, rng)
        syndrome = cal_syndrome(error, stabilizers)
        # print(error)
        for j, decoder in enumerate(decoders):
            t0 = time.perf_counter()
            recovery = decoder.decode(syndrome, error_prob)
            ts[j] += time.perf_counter() - t0
            success = decoding_success(lx, lz, recovery, error)
            if not success:
                fail_num[j] += 1
    return [r/trial_num for r in fail_num], ts

parser = argparse.ArgumentParser()
parser.add_argument("-code_type", type=str, default='sc')
parser.add_argument("-error_type", type=str, default='depolarizing')
parser.add_argument("-d_start", type=int, default=3)
parser.add_argument("-d_end", type=int, default=5)
parser.add_argument("-error_start", type=float, default=0.1)
parser.add_argument("-error_end", type=float, default=0.15)
parser.add_argument("-error_interval", type=int, default=2)
parser.add_argument("-trial", type=int, default=2)
args = parser.parse_args()

code_type = args.code_type
ds = np.arange(args.d_start, args.d_end, 2)
trial_num = args.trial
error_type = 'depolarizing'
error_rates = np.linspace(args.error_start, args.error_end, args.error_interval)
print(error_rates)
data = {}.fromkeys(ds)
for d in ds:
    data[d] = [[], [], [], []]
    if code_type == 'sc':
        n, m, hx, hz, lx, lz = gen_surface_code(d)
    elif code_type == 'rsc':
        n, m, hx, hz, lx, lz = gen_rotated_surface_code(d)
    assert not (hx @ lz.T % 2).any()
    assert not (hz @ lx.T % 2).any()
    stabilizers = block_diag(*(hz, hx))
    planar_decoder = Planar(hx, hz, lx, lz)
    # print(planar_decoder.pure_error_basis.shape)
    matching_decoder = pymatching_decoder_warpper(hx, hz)
    decoders = [matching_decoder, planar_decoder]
    
    # multiprocessing
    args = [
        (decoders, error_type, error_rate, trial_num, stabilizers, lx, lz) for error_rate in error_rates
    ]
    p = mp.Pool(len(error_rates))
    results = p.starmap(decode_subroutine, args)
    p.close()
    print(' d', '  er', '    lmw', '   lpl', '    tmw', '    tpl')
    for i, error_rate in enumerate(error_rates):
        print(
            f'{d:2d} {error_rate:.3f} | ' + 
            ('{:.4f} '*len(decoders)).format(*[r for r in results[i][0]]) +
            ('{:6.2f} '*len(decoders)).format(*[r for r in results[i][1]])
        )
        data[d][0].append(results[i][0][0])
        data[d][1].append(results[i][0][1])
        data[d][2].append(results[i][1][0])
        data[d][3].append(results[i][1][1])
print(data)

    # single thread
    # for error_rate in error_rates:
    #     if error_type == 'depolarizing':
    #         error_prob = np.array([1-error_rate, error_rate/3, error_rate/3, error_rate/3])
    #     elif error_type == 'z':
    #         error_prob = np.array([1-error_rate, 0.0, error_rate, 0.0])
    #     elif error_type == 'x':
    #         error_prob = np.array([1-error_rate, error_rate, 0.0, 0.0])
    #     elif error_type == 'uncorrelated':
    #         error_prob = np.array([1-error_rate, error_rate/3, error_rate/3, error_rate/3])
    #     else:
    #         raise ValueError('Unkown error type.')
    #     rng = np.random.default_rng(int(1000*error_rate))
    #     matching_decoder._set_up_decoder(error_prob)
    #     fail_num = [0] * len(decoders)
    #     ts = [0] * len(decoders)

    #     for trial in range(trial_num):
    #         error = generate_error(n, error_prob, rng)
    #         syndrome = cal_syndrome(error, stabilizers)
    #         # print(error)
    #         for j, decoder in enumerate(decoders):
    #             t0 = time.perf_counter()
    #             recovery = decoder.decode(syndrome, error_prob)
    #             ts[j] += time.perf_counter() - t0
    #             success = decoding_success(lx, lz, recovery, error)
    #             if not success:
    #                 fail_num[j] += 1
    #     print(
    #         f'{d:2d} {error_rate:.3f} | ' + 
    #         ('{:.4f} '*len(fail_num)).format(*[r/trial_num for r in fail_num]) +
    #         ('{:10.2f} '*len(fail_num)).format(*[r for r in ts])
    #     )
