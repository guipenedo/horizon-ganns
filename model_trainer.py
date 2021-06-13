from models.cgan_hdcas_spaces import *

torch.cuda.empty_cache()
train_model(10)
gan_model.save_model()
gan_model.save_losses()
