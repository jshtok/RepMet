def bb_overlap(bb_array, bb_GT):
    import numpy as np

    # bb_GT: [Ninstances,4] matrix
    # bb_array: [Nrois, 4]
    # for every row
    if bb_GT.ndim == 1:
        bb_GT = np.expand_dims(bb_GT,0)
    overlaps = np.zeros((bb_array.shape[0], bb_GT.shape[0]), np.float32)  # [Nrois, Ninstances]
    for i,GT in enumerate(bb_GT):  # go over rows
        # intersection
        ixmin = np.maximum(bb_array[:, 0], GT[0])  # [Nrois, 1]
        iymin = np.maximum(bb_array[:, 1], GT[1])
        ixmax = np.minimum(bb_array[:, 2], GT[2])
        iymax = np.minimum(bb_array[:, 3], GT[3])
        iw = np.maximum(ixmax - ixmin + 1., 0.)
        ih = np.maximum(iymax - iymin + 1., 0.)
        inters = iw * ih # [Nrois, 1]

        # union
        uni = ((GT[2] - GT[0] + 1.) * (GT[3] - GT[1] + 1.) +
               (bb_array[:, 2] - bb_array[:, 0] + 1.) *
               (bb_array[:, 3] - bb_array[:, 1] + 1.) - inters)

        overlaps[:, i] = inters / uni

    return overlaps