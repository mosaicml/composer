import composer.functional as cf
from torchvision import models

my_model = models.resnet18()

# add blurpool and squeeze excite layers
cf.apply_blurpool(my_model)
cf.apply_squeeze_excite(my_model)

# your own training code starts here
