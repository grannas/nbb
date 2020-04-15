import torch
from PIL import Image
import torchvision.transforms as transforms
from torch.autograd import Variable as V
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans


def read_image(path):
    preprocess = transforms.Compose([transforms.Resize(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                          std=[0.229, 0.224, 0.225])])

    image = Image.open(path).convert('RGB')
    processImage = V(preprocess(image).unsqueeze(0))
    return processImage, image


def vis_activations(tensor):
    array = tensor.detach().numpy()
    pic = np.sum(array, axis=1)[0]
    plt.imshow(pic)
    plt.show()


def tensor2pic(ImA):
    ImA = ImA[0].numpy()
    ImA = ImA.transpose(1, 2, 0)
    img = Image.fromarray(np.uint8(ImA * 255), 'RGB')
    img.show()


def compute_norm_act_map(F):
    F_norm = torch.norm(F, p=2, dim=1)[0]
    max_n = torch.max(F_norm)
    min_n = torch.min(F_norm)
    return (F_norm - min_n) / (max_n - min_n)


def vis_norm(tensor):
    pic = tensor.detach().numpy()
    plt.imshow(pic, cmap='jet')
    plt.show()


def kronecker(matrix1, matrix2):
    return torch.ger(matrix1.view(-1), matrix2.view(-1)).reshape(*(matrix1.size() + matrix2.size())).permute(
        [0, 2, 1, 3]).reshape(matrix1.size(0) * matrix2.size(0), matrix1.size(1) * matrix2.size(1))


def test_act(F):
    # This can be used for calculating V(p,q) at the last step
    H = []
    final_mapping = torch.zeros(224, 224)
    for f_l in F:
        H.append(compute_norm_act_map(f_l))

    for l in range(len(H)):
        act_map = H[l]
        for i in range(l):
            act_map = kronecker(act_map, torch.ones(2, 2))
        # vis_norm(act_map)
        final_mapping += act_map

    vis_norm(final_mapping)


def extract_from_clusters(nbbs, k):
    if len(nbbs) <= k:
        return nbbs

    # Converts to np TODO: Could maybe be done in torch?
    test = np.array([torch.cat((i[0], i[1]), 0).numpy() for i in nbbs])
    print(test)
    kmeans = KMeans(n_clusters=k, random_state=0).fit(test)
    print(kmeans.labels_)


def search_windows_extract(px, py, r, max_xy):
    # TODO: Cannot handle border cases, what to do? Use padding or force search radius to be above 0 and below max size?
    x_low = int(max(2 * px - r / 2, 0))
    x_high = int(min(2 * px + r / 2, max_xy))

    y_low = int(max(2 * py - r / 2, 0))
    y_high = int(min(2 * py + r / 2, max_xy))

    return (x_low, x_high), (y_low, y_high)


# Visualizing the NBB locations
def vis_nbb(layer):
    actA = layer.FA.detach().numpy()
    actB = layer.FB.detach().numpy()

    picA = np.sum(actA, axis=1)[0]
    plt.imshow(picA)
    for nbb in layer.nbbs:
        x = nbb[0][0]
        y = nbb[0][1]
        plt.scatter(x, y, s=15, c='red', marker='x')
    for nbb in layer.k_nbbs:
        x = nbb[0][0]
        y = nbb[0][1]
        plt.scatter(x, y, s=15, marker='o')
    plt.show()

    picB = np.sum(actB, axis=1)[0]
    plt.imshow(picB)
    for nbb in layer.nbbs:
        x = nbb[1][0]
        y = nbb[1][1]
        plt.scatter(x, y, s=15, c='red', marker='x')
    for nbb in layer.k_nbbs:
        x = nbb[1][0]
        y = nbb[1][1]
        plt.scatter(x, y, s=15, marker='o')
    plt.show()
