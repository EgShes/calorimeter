import torch
import torch.nn as nn
from businesslogic.net_utils import get_model
import pickle


class IngredientsModel(nn.Module):

    def __init__(self, args, model_weights_path, ingredients_vocab_path, instructions_vocab_path, transforms, device):
        super().__init__()
        self.device = device

        self.ingred_vocab = pickle.load(open(ingredients_vocab_path, 'rb'))
        self.instr_vocab = pickle.load(open(instructions_vocab_path, 'rb'))
        self.transforms = transforms

        self.model = get_model(**args).to(self.device)
        self.model.load_state_dict(torch.load(model_weights_path, map_location='cpu'))

    def forward(self, image):
        outs = self.model.sample(image, device=self.device, greedy=False)
        return outs
