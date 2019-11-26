
'''----------------------------------------------------------------------------
Main analysis for the conus project
----------------------------------------------------------------------------'''
import os
import time
import conusfun as cfun
import pickle

def analyze_cell_wrapper(arg):
    args, kwargs = arg
    return cfun.analyze_cell(*args, **kwargs)

if __name__ == '__main__':
    init_time = time.time()

    INPUT = pickle.load( open( os.path.join(
                                cfun.pickletemp, "inputlist.p"), "rb" ) )
    ninput = len(INPUT)
    # export the size of INPUT here

    jobindex = int(os.environ['SLURM_ARRAY_TASK_ID'])

    # skipdone = False ### by default rewrite them all!!
    # already_done = os.listdir(cfun.pickletemp)
    # if not skipdone or 'resdict_{}.p'.format(jobindex) not in already_done:

    resdict = analyze_cell_wrapper(INPUT[jobindex])

    pickle.dump( resdict, open( os.path.join(
                    cfun.pickletemp, "resdict_{}.p"
                    .format(jobindex)), "wb" ) )

    with open(os.path.join(cfun.outdir_data, 'jobindex.txt'), 'a') as file1:
        file1.write('running job index = {}\n'.format(jobindex))
        for item in list(resdict.keys()):
            file1.write(' {} '.format(item))
        file1.write('\n')

