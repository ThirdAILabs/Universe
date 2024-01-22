from thirdai import smx
import pytest

pytestmark = [pytest.mark.unit]


class CustomModule(smx.Module):
    def __init__(self, dim, submodule):
        super().__init__()
        self.layer = smx.Linear(dim, 2 * dim)
        self.submodule = submodule

    def forward(self, x):
        pass


def test_parameter_discovery():
    mod1 = CustomModule(dim=4, submodule=None)
    mod2 = CustomModule(dim=5, submodule=mod1)
    mod3 = CustomModule(dim=6, submodule=mod2)

    params = mod3.parameters()

    assert len(params) == 6
    assert mod1.layer.weight in params
    assert mod1.layer.bias in params
    assert mod2.layer.weight in params
    assert mod2.layer.bias in params
    assert mod3.layer.weight in params
    assert mod3.layer.bias in params


def test_cannot_register_module_with_itself():
    mod = CustomModule(dim=4, submodule=None)

    with pytest.raises(RuntimeError, match="Cannot register a module with itself."):
        mod.submodule = mod


def test_cannot_register_module_with_child():
    mod1 = CustomModule(dim=4, submodule=None)
    mod2 = CustomModule(dim=5, submodule=mod1)
    mod3 = CustomModule(dim=6, submodule=mod2)

    with pytest.raises(
        RuntimeError,
        match="Cannot register module as it contains the module it is being "
        "registered with as a submodule.",
    ):
        mod1.submodule = mod3
