import time
import torch
start_cuda = torch.cuda.Event(enable_timing=True, blocking=True)
finish_cuda = torch.cuda.Event(enable_timing=True, blocking=True)

def TicTocGenerator():
    # Generator that returns time differences
    ti = 0           # initial time
    tf = time.time() # final time
    while True:
        ti = tf
        tf = time.time()
        yield tf-ti # returns the time difference

TicToc = TicTocGenerator() # create an instance of the TicTocGen generator

# This will be the main function through which we define both tic() and toc()
def toc(pre_string='', verbose=True):
    # Prints the time difference yielded by generator instance TicToc
    tempTimeInterval = next(TicToc)
    if verbose:
        if pre_string != '':
            print(pre_string + ": %f seconds." % tempTimeInterval)
        else:
            print("Elapsed time: %f seconds." %tempTimeInterval )


def get_toc(pre_string='', verbose=True):
    # Prints the time difference yielded by generator instance TicToc
    tempTimeInterval = next(TicToc)
    if verbose:
        if pre_string != '':
            print(pre_string + ": %f seconds." % tempTimeInterval)
        else:
            print("Elapsed time: %f seconds." %tempTimeInterval )
    return tempTimeInterval

def tic():
    # Records a time in TicToc, marks the beginning of a time interval
    toc('', False)

def gtic():
    torch.cuda.synchronize()
    start_cuda.record()

def gtoc(pre_string='', verbose=True):
    ### Print: ###
    finish_cuda.record()
    torch.cuda.synchronize()
    total_time = start_cuda.elapsed_time(finish_cuda)
    if pre_string != '':
        print(pre_string + ": %f msec." % total_time)
    else:
        print("Elapsed time: %f msec." % total_time)
    return total_time





