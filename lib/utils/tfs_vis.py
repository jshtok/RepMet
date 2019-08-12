import os
import numpy as np
import cPickle
from sklearn.manifold import TSNE

def get_reps_proj(reps,samples,pars):
    indices_reps = []
    indices_samples = []
    X = np.zeros((pars.Edim, 0))
    for idx in range(pars.Nclasses):
        reps_c = reps[:, :, idx]
        samples_c = samples[idx]
        endp = X.shape[1]
        X = np.concatenate((X, reps_c, samples_c), axis=1)
        indices_reps += [range(endp, endp + reps_c.shape[1])]
        indices_samples += [range(endp + reps_c.shape[1], endp + reps_c.shape[1] + samples_c.shape[1])]
    X_embedded = TSNE(n_components=2, random_state=0, n_iter=500).fit_transform(X.transpose())

    return X_embedded, indices_reps, indices_samples

def get_bg_reps_proj(bg_reps,bg_samples,pars):
    Nbg_reps = bg_reps.shape[1]
    X = np.concatenate((bg_reps, bg_samples), axis=1)
    indices_reps = range(Nbg_reps)
    indices_samples = range(Nbg_reps, X.shape[1])
    X_embedded = TSNE(n_components=2, random_state=0, n_iter=500).fit_transform(X.transpose())
    return X_embedded, indices_reps, indices_samples

def normalize_reps(reps):
    reps_norm = np.sqrt(np.sum(np.square(reps), axis=0))
    if reps.ndim == 3:
        for i in range(reps.shape[1]):
            for j in range(reps.shape[2]):
                reps[:, i, j] = reps[:, i, j] / reps_norm[i, j]
    if reps.ndim == 2:
        for i in range(reps.shape[1]):
            reps[:, i] = reps[:, i] / reps_norm[i]
    return reps

def vis_reps_TSNE(samples,reps,bg_samples,bg_reps,pars):
    # inputs:
    # par: Edim, REP_L2_NORM, Nreps, Nbg_reps, Nclasses, dpi_value = 1200, pars.GroupSize = 8
    # samples, reps1, reps2, bg_reps1, bg_reps2.  # reps=[Edim, Nreps, Nclasses] bg_reps = [Edim, Nbg_reps], samples = [ [Edim, N] ...] in a list of length Nclasses corresp. to order of reps

    import matplotlib.pyplot as plt
    import matplotlib.colors as colors
    import matplotlib.cm as cmx

    if pars.REP_L2_NORM:
        reps = normalize_reps(reps)
        if pars.do_BG:
            bg_reps = normalize_reps(bg_reps)

    # print the reps =======================================================================================

    X_embedded, indices_reps, indices_samples = get_reps_proj(reps,samples,pars)

    # with open(pars.X_embedded_fname, 'wb') as fid:
    #     cPickle.dump({'X_embedded': X_embedded, 'indices_reps': indices_reps}, fid, protocol=cPickle.HIGHEST_PROTOCOL)
    # with open('/dccstor/jsdata1/dev/RepMet/vis_data.pkl','wb') as fid:
    #     cPickle.dump({'reps':reps,'reps_mat':reps_mat,'bg_reps_mat':bg_reps_mat,'bg_reps':bg_reps,'samples':samples,'bg_embeds':bg_embeds},fid,protocol=cPickle.HIGHEST_PROTOCOL)

    Ngroups = int(np.ceil(pars.Nclasses / pars.GroupSize))
    jet = cm = plt.get_cmap('jet')
    cNorm = colors.Normalize(vmin=0, vmax=pars.GroupSize)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
    for gn in range(Ngroups):
        save_fname = os.path.join(pars.vis_reps_fname_pref + '{0}.jpg'.format(gn))
        fig, ax = plt.subplots(1, 1)
        # for idx in range(pars.Nclasses):
        for idx in range(gn * pars.GroupSize, (gn + 1) * pars.GroupSize):
            if idx >= pars.Nclasses:
                break
            colorVal = scalarMap.to_rgba(idx - gn * pars.GroupSize)
            ax.scatter(X_embedded[indices_reps[idx], 0], X_embedded[indices_reps[idx], 1], marker='x', s=7, color=colorVal)
            ax.scatter(X_embedded[indices_samples[idx], 0], X_embedded[indices_samples[idx], 1], color=colorVal, s=0.15)
        ax.axis('off')
        fig.savefig(save_fname, dpi=pars.dpi_value)
        plt.close(fig)

    cNorm = colors.Normalize(vmin=0, vmax=pars.Nclasses)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
    save_fname = pars.vis_reps_fname_pref+'_all.jpg'
    fig, ax = plt.subplots(1, 1)
    for idx in range(pars.Nclasses):
        colorVal = scalarMap.to_rgba(idx)
        ax.scatter(X_embedded[indices_reps[idx], 0], X_embedded[indices_reps[idx], 1], marker='x', s=7, color=colorVal)
        ax.scatter(X_embedded[indices_samples[idx], 0], X_embedded[indices_samples[idx], 1], color=colorVal, s=0.15)
    ax.axis('off')

    fig.savefig(save_fname, dpi=pars.dpi_value)
    plt.close(fig)
    # print the bg_reps =======================================================================================
    if pars.do_BG:
        X_embedded, indices_reps, indices_samples = get_bg_reps_proj(bg_reps,bg_samples,pars)
        save_fname = pars.vis_bg_reps_fname
        fig, ax = plt.subplots(1, 1)
        ax.scatter(X_embedded[indices_samples, 0], X_embedded[indices_samples, 1], color=[0, 1, 0], s=0.15)
        ax.scatter(X_embedded[indices_reps, 0], X_embedded[indices_reps, 1], marker='x', s=7, color=[1, 0, 0])
        ax.axis('off')
        fig.savefig(save_fname, dpi=pars.dpi_value)
        plt.close(fig)
