import torch


def channel_stats(Im):
    epsilon = 1e-10  # Eps makes sure the std is not zero, which will break computations later
    mu = torch.mean(Im[0], dim=(1, 2))
    sigma = torch.zeros(mu.shape)
    for i in range(len(mu)):
        # TODO: Rewrite this w/o loop
        sigma[i] = torch.pow(torch.mean(torch.pow(torch.add(Im[0, i, :, :], -mu[i]), 2)) + epsilon, 0.5)
    return mu, sigma


def add_style(Im, muIm, sigmaIm, mu_mean, sigma_mean, fitToRange=False):
    newIm = torch.zeros(Im.shape)
    for i in range(len(mu_mean)):
        newIm[0, i, :, :] = (Im[0, i, :, :] - muIm[i]) / sigmaIm[i] * sigma_mean[i] + mu_mean[i]
        if fitToRange:
            newIm[0, i, :, :][newIm[0, i, :, :] < 0] = 0
            newIm[0, i, :, :][newIm[0, i, :, :] > 255] = 255
        if torch.max(torch.isnan(newIm)) == 1:
            print("Found NaN in add_style. Consider adjusting epsilon.")
    return newIm


def style_transfer(ImA, ImB):
    """
    Used on pictures the result is quite horrible
    I assume it works better if one starts at layer five and works up from there
    Should be able to handle any amount of channels
    """
    muA, sigmaA = channel_stats(ImA)
    muB, sigmaB = channel_stats(ImB)
    mum = (muA + muB) / 2
    sigmam = (sigmaA + sigmaB) / 2

    CA = add_style(ImA, muA, sigmaA, mum, sigmam)
    CB = add_style(ImB, muB, sigmaB, mum, sigmam)

    return CA, CB
