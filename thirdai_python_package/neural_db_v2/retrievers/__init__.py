from core.retriever import Retriever
from retrievers.mach import Mach
from retrievers.mach_mixture import MachMixture


def retriever_by_name(name: str, **kwargs):
    if name == "default":
        return Mach(**kwargs)
    if name == "mach":
        return Mach(**kwargs)
    if name == "machmixture":
        return MachMixture(**kwargs)
    raise ValueError(f"Invalid retriever name {name}")
