from models.cgan_water import *

torch.cuda.empty_cache()
train_model(20)
gan_model.save_model()
gan_model.save_losses()
