
import os
import glob
import imageio

import torch
from torchvision import models
from torch import optim

import matplotlib.pyplot as plt

import helpers

# Using VGG as other research papers have shown success in style transfers with
# the network. Will be using it for transfer learning
model = models.vgg19(models.VGG19_Weights.IMAGENET1K_V1).features

# Freeze current parameters of model
for param in model.parameters():
    param.requires_grad_(False)

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)

content = helpers.load_image('images/cat3.jpg').to(device)
style = helpers.load_image(
    'images/wave.jpg', shape=content.shape[-2:]).to(device)

# display image and style
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
ax1.imshow(helpers.image_convert(content))
ax2.imshow(helpers.image_convert(style))
plt.show()

content_features = helpers.get_features(content, model)
style_features = helpers.get_features(style, model)

# calculate gram matrix for each layer in the style features
style_grams = {layer: helpers.gram_matrix(
    style_features[layer]) for layer in style_features}

# initial target image initilized as content image
target = content.clone().requires_grad_(True).to(device)

# change these to pioritize differnt parts of the style
# layers closer to input are larger style features, smaller near output.
style_weights = {'conv1_1': 1.,
                 'conv2_1': 0.8,
                 'conv3_1': 0.5,
                 'conv4_1': 0.6,
                 'conv5_1': 0.1}


# weight losses, less stylization would mean less of the style weight.
CONTENT_WEIGHT = 1  # alpha
STYLE_WEIGHT = 1e8  # beta

# when to display style progress
SHOW_EVERY = 100

# creates a gif by saving img every "save_every" steps
MAKE_GIF = True
SAVE_EVERY = 100

optimizer = optim.Adam([target], lr=0.003)
STEPS = 1000

# pass image through "steps" times
for ii in range(1, STEPS+1):

    target_features = helpers.get_features(target, model)
    content_loss = torch.mean(
        (target_features['conv4_2'] - content_features['conv4_2'])**2)

    style_loss = 0

    # iterate through each style layer and add to the style loss
    for layer in style_weights:
        # get the "target" style representation for the layer
        target_feature = target_features[layer]
        _, d, h, w = target_feature.shape

        target_gram = helpers.gram_matrix(target_feature)

        style_gram = style_grams[layer]
        layer_style_loss = style_weights[layer] * \
            torch.mean((target_gram - style_gram)**2)

        # running style loss
        style_loss += layer_style_loss / (d * h * w)

    total_loss = CONTENT_WEIGHT * content_loss + STYLE_WEIGHT * style_loss

    # update your target image
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    # display intermediate images and print the loss
    if ii % SHOW_EVERY == 0:
        print('Total loss: ', total_loss.item())
        plt.imshow(helpers.image_convert(target))
        plt.tick_params(
            axis='both',
            which='both',
            bottom=False,
            top=False,
            left=False,
            labelleft=False,
            labelbottom=False)
        plt.show()

    # saves imgs to be used to make a gif later
    if (MAKE_GIF and ii % SAVE_EVERY == 0):
        plt.imshow(helpers.image_convert(target))
        plt.tick_params(
            axis='both',
            which='both',
            bottom=False,
            top=False,
            left=False,
            labelleft=False,
            labelbottom=False)
        plt.savefig('./gifs/' + str(ii) + '.png')

# creates gif and deletes imgs afterward
if MAKE_GIF:
    os.chdir("./gifs")
    files = sorted(glob.glob("*.png"), key=os.path.getmtime)
    with imageio.get_writer('mygif.gif', mode='I') as writer:
        for filename in files:
            image = imageio.v2.imread(filename)
            writer.append_data(image)
    for f in files:
        os.remove(f)
    os.chdir("../")
