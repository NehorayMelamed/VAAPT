import numpy
def my_linspace(start,stop,number_of_steps):
    bla = numpy.arange(start, stop, stop/number_of_steps);
    return bla

def my_logspace(start,stop,number_of_steps,log_base=10):
    start = 0;
    stop = 10;
    number_of_steps = 10
    log_base = 10;
    numpy.logspace( log10(start), log10(stop), num=number_of_steps, base=log_base)
