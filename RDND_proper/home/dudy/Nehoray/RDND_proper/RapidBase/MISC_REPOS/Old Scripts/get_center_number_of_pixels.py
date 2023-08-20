# from importlib import reload
# import things_to_import
# things_to_import = reload(things_to_import)
# from things_to_import import *

from numpy.fft import *
from numpy.linalg import *
from numpy import *
from numpy.random import *
from numpy import power as power
from numpy import exp as exp
start = -1
end = -1
def get_center_number_of_pixels(mat_in,number_of_center_pixels):   
    mat_in_shape = numpy.shape(mat_in);
    mat_in_rows = mat_in_shape[0];
    mat_in_cols = mat_in_shape[1];
    mat_in_rows_excess = mat_in_rows - number_of_center_pixels;
    mat_in_cols_excess = mat_in_cols - number_of_center_pixels;
    return mat_in[int(start + mat_in_rows_excess/2) : int(end-mat_in_rows_excess/2) , \
                  int(start + mat_in_cols_excess/2) : int(end-mat_in_cols_excess/2)];



