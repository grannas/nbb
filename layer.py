import torch
import numpy as np
from sklearn.cluster import KMeans

from utils import compute_norm_act_map, kronecker
from AdaIN import style_transfer


class Layer:
    def __init__(self, FA, FB, neigh, patch, padding, k):
        self.FA = FA
        self.FB = FB
        self.regions = []
        self.nbbs = []
        self.k_nbbs = []
        self.HA = compute_norm_act_map(FA)
        self.HB = compute_norm_act_map(FB)
        self.VA = None
        self.VB = None
        self.k = k
        self.neigh = neigh
        self.patch = patch
        self.padding = padding

        self.v_list = []
        self.k_v_list = []

        self.useNaiveRankCalc = False

    def filter_k_nbbs(self):

        if len(self.nbbs) > self.k:
            # Converts to np TODO: Could maybe be done in torch?
            nbbs = np.array([torch.cat((i[0], i[1]), 0).numpy() for i in self.nbbs], dtype=int)
            kmeans = KMeans(n_clusters=self.k, random_state=0).fit(nbbs)

            best_nbb_V = [-1] * self.k
            best_nbb_idx = [-1] * self.k

            for i, label in enumerate(kmeans.labels_):
                # Using the legacy rank calculation system (very similar, but not sure which is used in paper)
                if self.useNaiveRankCalc:
                    nbb_V = float(self.VA[nbbs[i, 1], nbbs[i, 0]] + self.VB[nbbs[i, 3], nbbs[i, 2]])
                else:
                    nbb_V = float(self.v_list[i])

                if nbb_V > best_nbb_V[label]:
                    best_nbb_V[label] = nbb_V
                    best_nbb_idx[label] = i

            for idx in best_nbb_idx:
                self.k_nbbs.append(self.nbbs[idx])
                self.k_v_list.append(self.v_list[idx])
        else:
            self.k_nbbs = self.nbbs
            self.k_v_list = self.v_list

        assert len(self.k_v_list) == len(self.k_nbbs), "Calculated V list length and k nbbs differ"

    def extract_nbbs(self):
        pairs = []
        v_list = []

        for region in self.regions:
            new_pairs = region.extract_nbb(self.FA, self.FB, self.HA, self.HB, self.patch, self.padding)
            new_vs = [region.rank] * len(new_pairs)

            for i, pair in enumerate(new_pairs):
                new_vs[i] += float(
                    self.HA[int(pair[0][1]), int(pair[0][0])] + self.HB[int(pair[1][1]), int(pair[1][0])])

            v_list += new_vs
            pairs += new_pairs

        self.nbbs = pairs
        self.v_list = v_list

    def calc_new_Vs(self, prev_VA=None, prev_VB=None, firstLayer=False):
        if firstLayer:
            self.VA, self.VB = self.HA, self.HB
        else:
            self.VA = self.HA + kronecker(prev_VA, torch.ones(2, 2))
            self.VB = self.HB + kronecker(prev_VB, torch.ones(2, 2))

    def generate_region(self, axrange, ayrange, bxrange, byrange, rank=0):
        new_region = region(axrange, ayrange, bxrange, byrange, rank)
        self.regions.append(new_region)


class region:
    # Region should inherit all relevant classes shit from layer?
    def __init__(self, px, py, qx, qy, rank=0):
        self.px = px
        self.py = py
        self.qx = qx
        self.qy = qy

        self.rank = rank

    def extract_nbb(self, FA, FB, HA, HB, patch, padding):
        pairs = determine_NN_pairs_2(FA, FB, HA, HB, self.px, self.py, self.qx, self.qy, padding, patch)

        return pairs

    def convert_to_layer_indices(self, pairs):
        # nbb pairs are given as region indices. Must be translated to the indices of the whole layer
        for i in range(len(pairs)):
            temp_pair = pairs[i]
            temp_pair[0][0] += self.px[0]
            temp_pair[0][1] += self.py[0]
            temp_pair[1][0] += self.qx[0]
            temp_pair[1][1] += self.qy[0]
            pairs[i] = temp_pair
        return pairs


def similarity(p, q, CA, CB, patch):
    d = torch.Tensor([[0.]])
    for neuron in patch:
        i = neuron + p
        j = neuron + q

        # Negative indeces leads to unwanted results
        if int(i[0]) < 0 or int(i[1]) < 0 or int(j[0]) < 0 or int(j[1]) < 0:
            continue

        # Try is here to ignore neighbours exceeding tensor range
        try:
            i_tensor = CA[:, :, int(i[1]), int(i[0])]
            i_tensor /= torch.norm(i_tensor, p=2)
            j_tensor = CB[:, :, int(j[1]), int(j[0])]
            j_tensor /= torch.norm(j_tensor, p=2)
            d += torch.mm(i_tensor, j_tensor.t())
        except:
            pass
    return d


def determine_NN_pairs_2(FA, FB, HA, HB, px, py, qx, qy, padding, patch):
    # Assuming square activations
    maxA = FA.shape[2] - 1
    maxB = FB.shape[2] - 1

    # Assures that padding do not exceed actual borders
    px_pad = (max(0, px[0] - padding), min(maxA, px[1] + padding))
    py_pad = (max(0, py[0] - padding), min(maxA, py[1] + padding))
    qx_pad = (max(0, qx[0] - padding), min(maxB, qx[1] + padding))
    qy_pad = (max(0, qy[0] - padding), min(maxB, qy[1] + padding))

    # Extract search window with added padding (for the similarity measure)
    PA = FA[:, :, py_pad[0]:py_pad[1] + 1, px_pad[0]:px_pad[1] + 1]
    QB = FB[:, :, qy_pad[0]:qy_pad[1] + 1, qx_pad[0]:qx_pad[1] + 1]

    # Generates common appearance of search windows
    CA, CB = style_transfer(PA, QB)

    HA_reg = HA[py_pad[0]:py_pad[1] + 1, px_pad[0]:px_pad[1] + 1]
    HB_reg = HB[qy_pad[0]:qy_pad[1] + 1, qx_pad[0]:qx_pad[1] + 1]

    pairs = []
    gamma = 0.05

    for i in range(py[0] - py_pad[0], py[1] - py_pad[0] + 1):
        for j in range(px[0] - px_pad[0], px[1] - px_pad[0] + 1):

            # p and q must be relevant
            if HA_reg[i, j] < gamma:
                continue

            # Finding the most similar neuron in Q for p
            p = torch.Tensor([j, i])
            best_neuron = torch.Tensor([0, 0])
            best_d = -1

            for k in range(qy[0] - qy_pad[0], qy[1] - qy_pad[0] + 1):
                for l in range(qx[0] - qx_pad[0], qx[1] - qx_pad[0] + 1):

                    # p and q must be relevant
                    if HB_reg[k, l] < gamma:
                        continue

                    q = torch.Tensor([l, k])
                    temp_d = similarity(p, q, CA, CB, patch)

                    if temp_d > best_d:
                        best_d = temp_d
                        best_neuron = q

            # Checking the found neuron q is also the most similar neuron to p in P
            best_buddies = True

            for i_ in range(py[0] - py_pad[0], py[1] - py_pad[0] + 1):
                for j_ in range(px[0] - px_pad[0], px[1] - px_pad[0] + 1):
                    p_ = torch.Tensor([j_, i_])
                    temp_d = similarity(p_, best_neuron, CA, CB, patch)

                    if temp_d > best_d:
                        best_buddies = False
                        break
                else:
                    continue
                break

            if best_buddies:
                pairs.append([p, best_neuron])

    # must remove the padding to get right indeces
    for i in range(len(pairs)):
        pairs[i][0][0] += px_pad[0]
        pairs[i][0][1] += py_pad[0]
        pairs[i][1][0] += qx_pad[0]
        pairs[i][1][1] += qy_pad[0]

    return pairs
