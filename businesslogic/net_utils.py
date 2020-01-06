import torch.nn as nn
from inversecooking.src.model import (EncoderLabels, EncoderCNN, DecoderTransformer, MaskedCrossEntropyCriterion,
                                      InverseCookingModel)
from torchvision import transforms


def get_transforms():
    tr = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))
    ])
    return tr


def get_model(embed_size=512, dropout_encoder=0.3, image_model='resnet50', dropout_decoder_r=0.3, maxseqlen=10,
              maxnuminstrs=1, n_att=8, transf_layers=16, dropout_decoder_i=0.3, maxnumlabels=20, n_att_ingrs=4,
              transf_layers_ingrs=4, ingrs_only=False, recipe_only=False, label_smoothing_ingr=0.1,
              ingr_vocab_size=1488, instrs_vocab_size=23231, **kwargs):
    # build ingredients embedding
    encoder_ingrs = EncoderLabels(embed_size, ingr_vocab_size,
                                  dropout_encoder, scale_grad=False)
    # build image model
    encoder_image = EncoderCNN(embed_size, dropout_encoder, image_model, pretrained=False)

    decoder = DecoderTransformer(embed_size, instrs_vocab_size,
                                 dropout=dropout_decoder_r, seq_length=maxseqlen,
                                 num_instrs=maxnuminstrs,
                                 attention_nheads=n_att, num_layers=transf_layers,
                                 normalize_before=True,
                                 normalize_inputs=False,
                                 last_ln=False,
                                 scale_embed_grad=False)

    ingr_decoder = DecoderTransformer(embed_size, ingr_vocab_size, dropout=dropout_decoder_i,
                                      seq_length=maxnumlabels,
                                      num_instrs=1, attention_nheads=n_att_ingrs,
                                      pos_embeddings=False,
                                      num_layers=transf_layers_ingrs,
                                      learned=False,
                                      normalize_before=True,
                                      normalize_inputs=True,
                                      last_ln=True,
                                      scale_embed_grad=False)
    # recipe loss
    criterion = MaskedCrossEntropyCriterion(ignore_index=[instrs_vocab_size - 1], reduce=False)

    # ingredients loss
    label_loss = nn.BCELoss(reduction=False)
    eos_loss = nn.BCELoss(reduction=False)

    model = InverseCookingModel(encoder_ingrs, decoder, ingr_decoder, encoder_image,
                                crit=criterion, crit_ingr=label_loss, crit_eos=eos_loss,
                                pad_value=ingr_vocab_size - 1,
                                ingrs_only=ingrs_only, recipe_only=recipe_only,
                                label_smoothing=label_smoothing_ingr)

    return model


def process_prediction(recipe_ids, ingredient_ids, recipe_vocab, ingredient_vocab):

    recipe = [recipe_vocab[tok] for tok in recipe_ids]
    if '<eoi>' in recipe:
        recipe = recipe[:recipe.index('<eoi>')] + [' .']
    title = ' '.join(recipe)

    ingredients = [ingredient_vocab[tok] for tok in ingredient_ids if ingredient_vocab[tok] != '<pad>']

    outs = {'title': title, 'ingredients': ingredients}

    return outs


def get_model_params():
    return dict(
        embed_size=512, dropout_encoder=0.3, image_model='resnet50', dropout_decoder_r=0.3, maxseqlen=10,
        maxnuminstrs=1, n_att=8, transf_layers=16, dropout_decoder_i=0.3, maxnumlabels=20, n_att_ingrs=4,
        transf_layers_ingrs=4, ingrs_only=False, recipe_only=False, label_smoothing_ingr=0.1,
        ingr_vocab_size=1488, instrs_vocab_size=23231
    )


def predict(img):
    pass