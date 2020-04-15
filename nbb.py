import torch
import torchvision.models as models
import matplotlib.pyplot as plt

from utils import read_image, search_windows_extract, vis_nbb
from model import VGG19
from layer import Layer


class neural_best_buddies:
    patch3 = torch.Tensor([[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]])
    patch5 = torch.Tensor(
        [[-2, -2], [-2, -1], [-2, 0], [-2, 1], [-2, 2], [-1, -2], [-1, -1], [-1, 0], [-1, 1], [-1, 2], [0, -2], [0, -1],
         [0, 1], [0, 2], [1, -2], [1, -1], [1, 0], [1, 1], [1, 2], [2, -2], [2, -1], [2, 0], [2, 1], [2, 2]])

    def __init__(self, fileA, fileB, k, k_n, s=0, e=13):
        self.nbb_pairs = {}
        self.R = {}

        self.IA, self.originalIA = read_image(fileA)
        self.IB, self.originalIB = read_image(fileB)
        self.model = VGG19(models.vgg19(pretrained=True))
        self.neighbourhoods = [4, 4, 4, 6, 6]
        self.patches = [self.patch5, self.patch5, self.patch5, self.patch3, self.patch3]
        self.padding = [2, 2, 2, 1, 1]
        self.K = [k, k * k_n, k * k_n, k * k_n, k * k_n]

        self.L = {}
        FA = self.model.forward(self.IA)
        FB = self.model.forward(self.IB)
        for l, F in enumerate(zip(FA, FB, self.neighbourhoods, self.patches, self.padding, self.K)):
            self.L[l + 1] = Layer(F[0], F[1], F[2], F[3], F[4], F[5])

        self.rnge = (s, e)

    def run_algorithm(self):
        self.L[5].generate_region(self.rnge, self.rnge, self.rnge, self.rnge)

        for l in range(5, 0, -1):
            print("layer:", l)
            if l == 5:
                self.L[l].calc_new_Vs(firstLayer=True)
            else:
                self.L[l].calc_new_Vs(self.L[l + 1].VA, self.L[l + 1].VB)

            self.L[l].extract_nbbs()
            if l == 1:
                self.L[l].nbbs = self.clean_nbbs(self.L[l].nbbs)
            self.L[l].filter_k_nbbs()

            if l > 1:
                for i, nbb in enumerate(self.L[l].k_nbbs):
                    px = int(nbb[0][0])
                    py = int(nbb[0][1])
                    qx = int(nbb[1][0])
                    qy = int(nbb[1][1])

                    a, b = search_windows_extract(px, py, self.L[l].neigh, self.L[l - 1].FA.shape[2] - 1)
                    c, d = search_windows_extract(qx, qy, self.L[l].neigh, self.L[l - 1].FB.shape[2] - 1)

                    self.L[l - 1].generate_region(a, b, c, d, self.L[l].k_v_list[i])

    def clean_nbbs(self, nbbs):
        LIMITS = [0, 1, 222, 223]
        cleaned_nbbs = []

        for pair in nbbs:
            cleaned_nbbs.append(pair)
            for tensor in pair:
                shouldBreak = False
                for limit in LIMITS:
                    if limit in tensor:
                        cleaned_nbbs.pop()
                        shouldBreak = True
                        break
                if shouldBreak:
                    break

        return cleaned_nbbs

    def vis_nbbs(self, layers=(5, 4, 3, 2, 1)):
        for l in layers:
            vis_nbb(self.L[l])

    def vis_final_nbbs(self, save=False, fileName=""):

        nbbs = self.L[1].k_nbbs

        plot_nbb_pic(self.originalIA, nbbs, 0, save, fileName)
        plot_nbb_pic(self.originalIB, nbbs, 1, save, fileName)

    def save_result(self, fileName):
        self.vis_final_nbbs(True)

    def print_layer_stats(self, layers=(5, 4, 3, 2, 1)):
        for l in layers:
            print("Layer:", l)
            print("Totalt antal:", len(self.L[l].nbbs))
            # print(L[l].nbbs)
            print("Sorterat antal", len(self.L[l].k_nbbs))
            # print(L[l].k_nbbs)
            print("Antal regioner", len(self.L[l].regions))


def plot_nbb_pic(pic, nbbs, idx, save=False, fileName=""):
    plt.imshow(pic)
    plt.axis('off')
    for nbb in nbbs:
        x = nbb[idx][0]
        y = nbb[idx][1]
        plt.scatter(x, y, marker='o', edgecolors='w', s=50)
    if save:
        plt.savefig('results/finalPic' + fileName + str(idx) + '.png', bbox_inches='tight')
        plt.show()
    else:
        plt.show()
