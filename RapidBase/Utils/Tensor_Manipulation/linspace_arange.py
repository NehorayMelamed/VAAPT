

import numpy
from numpy import arange
def matlab_arange(start,step,stop):
    return arange(start,stop,step);

def my_linspace(start,stop,number_of_steps):
    bla = numpy.arange(start, stop, (stop-start)/number_of_steps);
    return bla

def my_linspace_step(start, step, number_of_steps):
    stop = start + step*number_of_steps
    if step == 0 and number_of_steps == 0:
        return None
    elif step == 0 and number_of_steps > 0:
        bla = numpy.array([start]*number_of_steps)
    else:
        bla = numpy.arange(start, start+step*number_of_steps, step)
    return bla

def my_linspace_step2(start, stop, step):
    return numpy.arange(start, stop, step)

def my_logspace(start,stop,number_of_steps,log_base=10):
    start = 0;
    stop = 10;
    number_of_steps = 10
    log_base = 10;
    numpy.logspace( log10(start), log10(stop), num=number_of_steps, base=log_base)


def my_linspace_int(start,step,number_of_steps):
    stop = start + number_of_steps*step;
    return arange(start,stop,step,int);


def int_arange(start,step,stop):
    return int(arange(start,stop,step))


def int_range(start,step,stop):
    return arange(start,stop,step,'int')


def mat_range(start,step,stop):
    return arange(start,stop,step);



