__all__ = ["HGNN", "HGNN_PLUS", "HNHN", "UNIGAT", "HyperGCN"]


def __getattr__(name):
    if name == "HGNN":
        from .hgnn import HGNN
        return HGNN
    if name == "HGNN_PLUS":
        from .hgnn_plus import HGNN_PLUS
        return HGNN_PLUS
    if name == "HNHN":
        from .hnhn import HNHN
        return HNHN
    if name == "UNIGAT":
        from .unigat import UNIGAT
        return UNIGAT
    if name == "HyperGCN":
        from .hypergcn import HyperGCN
        return HyperGCN
    raise AttributeError(name)
