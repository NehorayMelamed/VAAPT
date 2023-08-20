from RDND_proper.models.EGVSR.codes.models.vsr_model import VSRModel
from RDND_proper.models.EGVSR.codes.models.vsrgan_model import VSRGANModel

from RDND_proper.models.EGVSR.codes.models.espcn_model import ESPCNModel
from RDND_proper.models.EGVSR.codes.models.vespcn_model import VESPCNModel
from RDND_proper.models.EGVSR.codes.models.sofvsr_model import SOFVSRModel

# register vsr model
vsr_model_lst = [
    'frvsr',
]

# register vsrgan model
vsrgan_model_lst = [
    'tecogan',
]


def define_model(opt):
    if opt['model']['name'].lower() in vsr_model_lst:
        model = VSRModel(opt)

    elif opt['model']['name'].lower() in vsrgan_model_lst:
        model = VSRGANModel(opt)

    elif opt['model']['name'].lower() == 'espcn':
        model = ESPCNModel(opt)

    elif opt['model']['name'].lower() == 'vespcn':
        model = VESPCNModel(opt)

    elif opt['model']['name'].lower() == 'sofvsr':
        model = SOFVSRModel(opt)

    else:
        raise ValueError('Unrecognized model: {}'.format(opt['model']['name']))

    return model
