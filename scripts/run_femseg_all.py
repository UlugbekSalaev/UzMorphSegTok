"""Run all FEMSeg-uz configurations and seeds in parallel, then the cross-validation of the main model.
Finished runs are skipped.

Usage:  python scripts/run_femseg_all.py [parallel processes] [threads per process] [what: all, main, cv]
"""
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C

GROUPS = [['full', 'v3', 'full_nocse', 'full_char'], ['v2', 'v3v2'], ['full_notrans', 'full_nocnn']]
FOLDS = 5


def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    threads = sys.argv[2] if len(sys.argv) > 2 else '2'
    what = sys.argv[3] if len(sys.argv) > 3 else 'all'
    logs = os.path.join(C.ROOT, 'logs')
    os.makedirs(logs, exist_ok=True)
    jobs = []
    if what in ('all', 'main'):
        jobs += [(c, s, None) for g in GROUPS for s in C.SEEDS for c in g]
    if what in ('all', 'cv'):
        jobs += [('full', C.SEEDS[0], k) for k in range(FOLDS)]
    running = []
    while jobs or running:
        running = [(p, n) for p, n in running if p.poll() is None]
        while jobs and len(running) < workers:
            cfg, seed, fold = jobs.pop(0)
            name = f'{cfg}_s{seed}' + (f'_fold{fold}' if fold is not None else '')
            res = os.path.join(C.RESULTS, 'femseg_cv' if fold is not None else 'femseg')
            os.makedirs(res, exist_ok=True)
            if os.path.exists(os.path.join(res, name + '.json')):
                continue
            log = open(os.path.join(logs, f'femseg_{name}.log'), 'w')
            cmd = [sys.executable, os.path.join(C.ROOT, 'scripts', 's04_train_femseg.py'), '--config', cfg,
                   '--seed', str(seed), '--threads', threads]
            if fold is not None:
                cmd += ['--fold', str(fold)]
            p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=C.ROOT)
            running.append((p, name))
            print('started:', name, flush=True)
        time.sleep(20)
    print('all done', flush=True)


if __name__ == '__main__':
    main()
