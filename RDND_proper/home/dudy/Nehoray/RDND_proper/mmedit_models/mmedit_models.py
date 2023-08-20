# MMedit models

import torch
try:
    from mmedit_models.FTVSR import FTVSR # classic mmcv takes 2 tries to import successfully or else it crushes...
except:
    pass
from mmedit_models.FTVSR import FTVSR
from mmedit_models.RealBasicVSR import RealBasicVSRNet

def mmcv_load_cp(model, cp_path):

    state_dict = torch.load(cp_path)['state_dict']
    new_dict = {}

    for i,key in enumerate(state_dict):
        if i > 0: # don't include step_counter key - it's not a parameter of the model...
            new_dict[key.split('generator')[1][1:]] = state_dict[key] # match the correct key name

    model.load_state_dict(new_dict)

def FTVSR_base(pretrained=True, checkpoint_path='', train_device=torch.device(1)):

    model = FTVSR().to(train_device)

    if pretrained == True:
        mmcv_load_cp(model, checkpoint_path)

    else:  # Loading our checkpoint
        model.load_state_dict(torch.load(checkpoint_path, map_location="cuda:" + str(train_device))['model_state_dict'])

    print('Loading FTVSR_base')

    return model

def FTVSR_base(pretrained=True, checkpoint_path='', train_device=torch.device(1)):

    model = RealBasicVSRNet().to(train_device)

    if pretrained == True:
        mmcv_load_cp(model, checkpoint_path)

    else:  # Loading our checkpoint
        model.load_state_dict(torch.load(checkpoint_path, map_location="cuda:" + str(train_device))['model_state_dict'])

    print('Loading FTVSR_base')

    return model





# start with small lr images [180,320]
# add 4x4 patches algorithm - see how they do it
# finetune train - 64x64 patches, reg charb loss,
# make dataset with different comprression ratios and also non-compressed with some probability
# make dataset by first downsampling and then do different compressions
# train on B_W drones


# download realbasicvsr cpts
# fix checkpoints
# run it on obth



# add basicvsr ++ as well
# download the ckecpoints
# run it on both


# fix ftsvr with exploding values -wtf?


# run the other losses that were interested

# get yoav to send me the deturblunce layer

