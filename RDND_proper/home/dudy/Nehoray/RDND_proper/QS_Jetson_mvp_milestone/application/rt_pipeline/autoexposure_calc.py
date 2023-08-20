"""
function newexpo = Calc_new_exposure_time(fr,max_gl,oldexpo,max_saturated_pixel,sat_range,expo_max)
% fr - input frame
% max_gl - max gray lvl.
% oldexpo - current exposure time
% max_saturated_pixel - allow only 1% of the pixels to be saturated
% sat_range - num of gl that considered as saturated
% expo_max - max exposure time

ghst = hist(double(fr(:)),0:max_gl); %
fr_sz = size(fr);
N_pxl = fr_sz(1)*fr_sz(2);
sat_pixels = sum(ghst(end+(-sat_range:0)));
if sat_pixels>max_saturated_pixel
    newexpo = oldexpo*(1-sat_pixels/N_pxl)^10;
elseif sat_pixels==0
    newexpo = min([oldexpo*max_gl/(max_gl-find(ghst(end:-1:1),1)) ...
        expo_max*.999]);
else
    newexpo = oldexpo;
end

if abs(newexpo/oldexpo-1)<1e-2
    newexpo = oldexpo;
end
"""
import numpy as np


def highest_non_zero_index(arr: np.array):
    # finds highest non zero index in array
    return np.where(arr)[0].max()


def calc_new_exposure_time(frame: np.array, current_exposure_time: float, max_gray_level: int, max_saturated_pixels: float, saturation_range: int, exposure_max: float):
    ghst = np.histogram(frame, np.arange(0, max_gray_level+1))
    num_pixels = frame.shape[-2] * frame.shape[-1]
    saturated_pixels = sum(ghst[saturation_range-1:])
    if saturated_pixels > max_saturated_pixels:
        return current_exposure_time * (1 - saturated_pixels / num_pixels) ** 10
    elif saturated_pixels == 0:
        highest_gray_value = highest_non_zero_index(ghst[0])  # ghst[0] is the histogram itself
        return min(current_exposure_time*max_gray_level/(max_gray_level-highest_gray_value), exposure_max*0.999)     #current_exposure_time * (1 - saturated_pixels / num_pixels) ** 10
    else:
        return current_exposure_time
