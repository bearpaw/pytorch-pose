import torch
from models import create_model

# import models

# # create_model()

# model_names = sorted(name for name in models.__dict__
#     if name.islower() and not name.startswith("__")
#     and callable(models.__dict__[name]))

# model = models.__dict__['create_model']()

# model.cuda()

# crit = torch.nn.MSELoss()

# inp = torch.autograd.Variable(torch.Tensor(6, 3, 256, 256)).cuda()

# out = model(inp)

# loss = 0
# for o in out:
#    loss += crit(o, o)


# print loss