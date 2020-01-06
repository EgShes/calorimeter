from businesslogic.net_utils import get_model_params, get_transforms
from businesslogic.create_nets import IngredientsModel
import os

DEVICE = os.getenv('DEVICE', 'cpu')

model_weights_path = 'weights/model.ckpt'
ingredients_vocab_path = 'weights/ingredients_vocab.pkl'
instructions_vocab_path = 'weights/instructions_vocab.pkl'
transforms = get_transforms()
params = get_model_params()

print('Loading model')
print(f'Device to run model at - \033[1m{DEVICE.upper()}\033[0m')
model = IngredientsModel(params, model_weights_path, ingredients_vocab_path, instructions_vocab_path,
                         transforms, DEVICE).eval()
print('Model loaded successfully')
