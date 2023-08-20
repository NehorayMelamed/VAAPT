import torch

import models.RIDNet.TestCode.code.utility as utility
import models.RIDNet.TestCode.code.data as data
import models.RIDNet.TestCode.code.model as model
import models.RIDNet.TestCode.code.loss as loss
from RDND_proper.models.RIDNet.TestCode.code.option import args
from RDND_proper.models.RIDNet.TestCode.code.trainer import Trainer
from torch.nn import DataParallel

torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)

if checkpoint.ok:
    loader = data.Data(args)
    model = model.Model(args, checkpoint)
    model = DataParallel(model)
    loss = loss.Loss(args, checkpoint) if not args.test_only else None
    t = Trainer(args, loader, model, loss, checkpoint)
    while not t.terminate():
        t.train()
        t.test()

    checkpoint.done()

