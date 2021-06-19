from models.wgan_dp_no_cond_hdcas import *

torch.cuda.empty_cache()
train_model(40)
gan_model.save_model()
gan_model.save_losses()
