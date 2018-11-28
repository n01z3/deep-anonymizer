import cv2
# import matplotlib.pyplot as plt 
import numpy as np
from PIL import Image

def extract_largest_component (image):
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=4)
    sizes = stats[:, -1]

    max_label = 1
    max_size = sizes[1]
    
    for i in range(2, nb_components):
        if sizes[i] > max_size:
            max_label = i
            max_size = sizes[i]

    img2 = np.zeros(output.shape, dtype=np.uint8)
    img2[output == max_label] = 255

    return img2

def preds2dets(probs, img_pil, class_names, threshold):

    classes = np.argmax(probs, 0).astype(np.uint8)
    num_classes = probs.shape[0]

    dets = {}
    fy, fx = float(img_pil.size[1])/classes.shape[0], float(img_pil.size[0]) / classes.shape[0]
    for i in range(num_classes):

        nc = np.sum(classes == i) 

        confidence = 0 if nc == 0 else probs[i][classes == i].sum() / nc
            
        # print(confidence, class_names[i])
        if confidence > threshold:

            t = (classes == i).astype(np.uint8) * 255
            if not(class_names[i] == 'Sunglasses' or class_names[i] == 'Shoes'):
                t = extract_largest_component(t)

            x, y, w, h = cv2.boundingRect(t)

            x, y, w, h = int(fx * x), int(fy * y), int(fx *(x + w)) - int(fx * x), int(fy * (y + h)) - int(fy * y)

            # cropped = Image.fromarray(np.array(img_pil)[y: y + h , x:x+w])
            # cropped.save(f'{class_names[i]}.png')

            dets[class_names[i]] = {'x': x,
                                    'y': y,
                                    'w': w,
                                    'h': h,
                                    'i': i}
    return dets


def preds2coloredseg(probs, img_pil, out_format):

    classes = np.argmax(probs, 0).astype(np.uint8)
    num_classes = probs.shape[0]


    if out_format == 'color':
        colors = discrete_cmap(num_classes)[:,:-1]
        classes_ = (colors[classes] * 255).astype(np.uint8)
    else:
        classes_ = classes

    img_to_save = Image.fromarray(classes_)

    if (img_to_save.size[0] != img_pil.size[0]) or (img_to_save.size[1] != img_pil.size[1]):
        img_to_save = img_to_save.resize(img_pil.size, Image.NEAREST)


    return img_to_save

def discrete_cmap(N, base_cmap='viridis'):
    """Create an N-bin discrete colormap from the specified input map"""

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return color_list # base.from_list(cmap_name, color_list, N)
