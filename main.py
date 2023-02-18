"""
Play ground to test the model functionality
"""

from sys import argv
from backgroundRemoverV1 import  load_seg_model, remove_background, plot_image

if __name__ == "__main__":

    image_path = argv[1]  # first argument. image path
    get_only_most_repeated_label = argv[2] == "True"
    print(get_only_most_repeated_label)
    feature_extractor, model = load_seg_model()  # loading default model
    image_wo_background , image_w_background = remove_background(model,feature_extractor,image_path,get_only_most_repeated_label)
    print(get_only_most_repeated_label)
    plot_image(image_w_background)
    plot_image(image_wo_background)