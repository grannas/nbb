import torch.nn as nn


class VGG19(nn.Module):
    def __init__(self, model):
        super(VGG19, self).__init__()
        layer_ids = [[0, 2], [2, 7], [7, 12], [12, 21], [21, 30]]
        self.layers = []
        for ids in layer_ids:
            self.layers.append(self.get_layer(model, ids[0], ids[1]))

    def forward(self, x, level=5, start_level=0):
        out = []
        for i in range(start_level, level):
            layer = self.layers[i]
            x = layer(x)
            out.append(x)
        return out

    def get_layer(self, model, start, end):
        layer = []
        features = next(model.children())
        for idx, module in enumerate(features.children()):
            if start <= idx < end:
                layer.append(module)
        return nn.Sequential(*layer)
