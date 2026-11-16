import numpy as np
import pytest
import torch

from human_tracking.osnet import osnet_x1_0
from human_tracking.osnet_ain import osnet_ain_x1_0
from human_tracking.reid_embedder import REID_MODELS, ReIDEmbedder, load_reid_state_dict


def _random_checkpoint(constructor, prefix=""):
    torch.manual_seed(0)
    state = constructor(num_classes=10, pretrained=False).state_dict()
    return {prefix + k: v.clone() for k, v in state.items()}


def test_strips_dataparallel_prefix_and_ignores_classifier_head():
    checkpoint = _random_checkpoint(osnet_ain_x1_0, prefix="module.")
    model = osnet_ain_x1_0(num_classes=1000, pretrained=False)
    load_reid_state_dict(model, checkpoint)
    loaded = model.state_dict()
    assert torch.equal(loaded["conv1.conv.weight"], checkpoint["module.conv1.conv.weight"])
    assert loaded["classifier.weight"].shape[0] == 1000  # head kept at the model's own size


def test_partial_checkpoint_is_rejected_not_silently_half_loaded():
    checkpoint = _random_checkpoint(osnet_x1_0)
    del checkpoint["conv5.conv.weight"]
    with pytest.raises(ValueError, match="does not match"):
        load_reid_state_dict(osnet_x1_0(num_classes=1000, pretrained=False), checkpoint)


def test_wrong_architecture_checkpoint_is_rejected():
    with pytest.raises(ValueError, match="does not match"):
        load_reid_state_dict(osnet_x1_0(num_classes=1000, pretrained=False), _random_checkpoint(osnet_ain_x1_0))


def test_unknown_model_name_is_rejected():
    with pytest.raises(ValueError, match="Unknown ReID model"):
        ReIDEmbedder(weights_path="/nonexistent.pth", model="resnet50")


@pytest.mark.parametrize("model", REID_MODELS)
def test_embedder_outputs_unit_vectors_and_none_for_empty_crops(tmp_path, model):
    constructor = {"osnet_x1_0": osnet_x1_0, "osnet_ain_x1_0": osnet_ain_x1_0}[model]
    path = tmp_path / "w.pth"
    torch.save(_random_checkpoint(constructor, prefix="module."), path)

    rng = np.random.default_rng(0)
    crops = [rng.integers(0, 255, (120, 50, 3), dtype=np.uint8), None, np.zeros((0, 0, 3), np.uint8)]
    raw = ReIDEmbedder(weights_path=str(path), model=model, normalize=False).embed(crops)
    normed = ReIDEmbedder(weights_path=str(path), model=model, normalize=True).embed(crops)

    assert raw[1] is None and raw[2] is None
    assert raw[0].shape == (512,)
    assert np.isclose(np.linalg.norm(raw[0]), 1.0, atol=1e-5)
    assert np.isclose(np.linalg.norm(normed[0]), 1.0, atol=1e-5)
    assert not np.allclose(raw[0], normed[0])  # the flag really changes the model input


def test_normalization_is_on_by_default(tmp_path):
    path = tmp_path / "w.pth"
    torch.save(_random_checkpoint(osnet_x1_0), path)
    assert ReIDEmbedder(weights_path=str(path)).normalize is True
