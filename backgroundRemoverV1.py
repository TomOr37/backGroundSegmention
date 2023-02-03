"""
First approach.
Semi-rule based background remover.
First we use pre-trained model to label the pixels to 150 categories.
Then, we transform each "background" labeled pixel (pre-defined) to black - and getting the background.
Another approach was to take to most repeated label that isn't count as
background('get_only_most_repeated_label' flag)  and only count those labled pixels as non-background.

The model preforms the best when the objects in the image are realated to training data-set : ADE20K.

This approach has some flaws, but doesnt need more data to fine-tune the model.
we will explore fine-tuning segformer model in 'backgroundRemoverV2'.

pretrained model paper : "https://arxiv.org/pdf/2105.15203.pdf"


"""


from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
from PIL import Image
import requests
from torch import nn
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

background_labels = []  # labels which  the model will counts as background.


def load_seg_model(model_name: str = "nvidia/segformer-b5-finetuned-ade-640-640"):
    feature_extractor = SegformerFeatureExtractor.from_pretrained(model_name)
    model = SegformerForSemanticSegmentation.from_pretrained(model_name)
    return feature_extractor, model


def remove_background(model, feature_extractor, image_path, get_only_most_repeated_label: bool):
    """

    :param model:
    :param feature_extractor:
    :param image_path:
    :param get_only_most_repeated_label:
    :return:
    """
    global background_labels
    image = Image.open(image_path)
    inputs = feature_extractor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits  # shape (batch_size, num_labels, height/4, width/4)

    # First, rescale logits to original image size
    upsampled_logits = nn.functional.interpolate(logits,
                                                 size=image.size[::-1],  # (height, width)
                                                 mode='bilinear',
                                                 align_corners=False)

    # Second, apply argmax on the class dimension(Getting labels for each pixel)
    seg = upsampled_logits.argmax(dim=1)[0]
    mask_seg = np.ones((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)  # height, width, 3
    background_mask_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
    if get_only_most_repeated_label:
        main_label = get_max_repeted_elem(seg)
        mask_seg[seg == main_label, :] = [0, 0, 0]
        background_mask_seg[seg == main_label, :] = [1, 1, 1]

    else:
        for index in background_labels:
            mask_seg[seg == index, :] = [0, 0, 0]
            background_mask_seg[seg == index, :] = [1, 1, 1]
    np_image = np.array(image)
    image_wo_background = np.multiply(np_image, mask_seg)
    image_w_background = np.multiply(np_image, background_mask_seg)
    return image_wo_background, image_w_background


def get_max_repeted_elem(_tensor):
    """
    Purpose: getting the most repeated element in a tensor not including background_labels
    :param _tensor: input tensor
    :return: max repeated element not including background_labels.
    """

    max_relevent_elem = torch.tensor(0)
    max_count = torch.tensor(0)
    ts_elm = torch.unique(_tensor).numpy()
    for elem in ts_elm:
        count_elm = torch.count_nonzero(_tensor == elem)
        if max_count < count_elm and elem not in background_labels:
            max_count = count_elm
            max_relevent_elem = elem
    return max_relevent_elem


def npImage2Image(np_img):
    return Image.fromarray(np_img)


def plot_image(img):
    plt.figure(figsize=(15, 10))
    plt.imshow(img)
    plt.show()


def init_background_labels():
    """
    Initialing background_labels by defaults values. The default values are derived from
    https://raw.githubusercontent.com/CSAILVision/sceneparsing/master/objectInfo150.csv - a csv
    which includes all the labels of the data
    that the default model : 'segformer-b5-finetuned-ade-640-640' was trained on.
    Some of the labels are annotated with 'Stuff == 1', labels which are more likely to be
    count as background.(Sky , wall , celling , etc..).

    """
    global background_labels
    url_path = "https://raw.githubusercontent.com/CSAILVision/sceneparsing/master/objectInfo150.csv"
    labels_file = pd.read_csv(url_path, index_col=False)
    background_labels = np.array(labels_file.query("`Stuff` == 1")["Idx"].tolist()) - 1
