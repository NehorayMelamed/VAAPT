import torch
try:
    from mmedit_models.FTVSR import FTVSR # classic mmcv takes 2 tries to import successfully or else it crushes...
except:
    pass
from mmedit_models.FTVSR import FTVSR

def mmcv_load_cp(model, cp_path):

    state_dict = torch.load(cp_path)['state_dict']
    new_dict = {}

    for i,key in enumerate(state_dict):
        if i > 0: # don't include step_counter key - it's not a parameter of the model...
            new_dict[key.split('generator')[1][1:]] = state_dict[key] # match the correct key name

    model.load_state_dict(new_dict)


# cp = '/home/omerl/rdnd/FTVSR_REDS.pth'
cp2 = '/home/omerl/rdnd/FTVSR_Vimeo90K.pth'
a = FTVSR()
print ('starting to load')
print (a.conv_hr.parameters().__next__().sum())
mmcv_load_cp(a, cp2)
print ('finished loading')
print (a.conv_hr.parameters().__next__().sum())


gpu = 3
a = a.to(gpu)
x = torch.randn(1,8,3,64,64).to(gpu)
x2 = torch.randn(1,8,3,180,320).to(gpu)
#
with torch.no_grad():
    y = a(x2)

