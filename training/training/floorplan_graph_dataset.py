import numpy as np
from .dataset import ImageFolderDataset
from .graph_utils import extract_graph_features

class FloorplanGraphDataset(ImageFolderDataset):
    """Image dataset that generates graph features for each floor plan."""
    def __init__(self, path, resolution=None, **super_kwargs):
        super().__init__(path=path, resolution=resolution, use_labels=False, **super_kwargs)
        sample = super()._load_raw_image(self._raw_idx[0])
        feats = extract_graph_features(sample.transpose(1, 2, 0))
        self._label_shape = [len(feats)]
        self._graph_features = [extract_graph_features(super()._load_raw_image(i).transpose(1,2,0)) for i in range(self._raw_shape[0])]

    def __getitem__(self, idx):
        img = self._load_raw_image(self._raw_idx[idx])
        if self._xflip[idx]:
            img = img[:, :, ::-1]
        label = self._graph_features[self._raw_idx[idx]]
        return img.copy(), label.astype(np.float32)

    def get_label(self, idx):
        return self._graph_features[self._raw_idx[idx]].astype(np.float32)
