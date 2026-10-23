"""Public model tests use only tiny synthetic fixtures and temporary artifacts."""

from dataclasses import replace
import json

import pytest
import sympy as sp
import torch

from ai.inference import CandidateScorer, save_artifact
from ai.model import ActionTransformer, ModelConfig, PolynomialTokenizer, serialize_candidate
from ai.model.tokenizer import action_from_data, expression_from_data, expression_to_data
from mathai import Action, State, apply_action, enumerate_actions

torch.set_num_threads(2)
x, y, z = sp.symbols("x y z")


@pytest.fixture
def candidate():
    state = State((x+y)**2, x**2+2*x*y+y**2)
    action = Action("expand_power")
    return serialize_candidate(state, action, apply_action(state, action)[0])


@pytest.fixture
def tiny():
    torch.manual_seed(7)
    config = ModelConfig(embedding_dim=16, attention_heads=2, encoder_layers=1,
                         feedforward_dim=32, max_sequence_length=2048)
    return ActionTransformer(config), PolynomialTokenizer(config.max_sequence_length)


def test_structural_serialization_is_exact_and_distinguishes_equivalent_forms():
    expression = sp.Rational(-13, 7)*x**12 + sp.Symbol("alpha_2")*y
    assert expression_from_data(expression_to_data(expression)) == expression
    assert expression_to_data((x+y)**2) != expression_to_data(x**2+2*x*y+y**2)
    symbol = sp.Symbol("[CURRENT]λ__import__('os')")
    assert expression_from_data(expression_to_data(symbol)) == symbol


def test_tokenizer_roundtrip_unicode_fields_and_fixed_vocabulary(candidate, tmp_path):
    tokenizer = PolynomialTokenizer()
    encoded = tokenizer.encode(candidate)
    assert tokenizer.decode(encoded) == candidate
    assert tokenizer.unknown_id not in encoded
    assert tokenizer.vocab_size == 264
    tokenizer.save(tmp_path/"tokenizer.json")
    assert PolynomialTokenizer.load(tmp_path/"tokenizer.json").encode(candidate) == encoded
    unicode_state = State((sp.Symbol("λ[CURRENT]")+y)**2, 0)
    unicode_action = Action("expand_power")
    record = serialize_candidate(unicode_state, unicode_action, apply_action(unicode_state, unicode_action)[0])
    assert tokenizer.decode(tokenizer.encode(record)) == record


def test_padding_and_masks(candidate):
    tokenizer = PolynomialTokenizer()
    other = State(z*(x+y)**2, 0)
    action = Action("expand_power", (other.expression.args.index((x+y)**2),))
    record = serialize_candidate(other, action, apply_action(other, action)[0])
    batch = tokenizer.batch_encode([candidate, record])
    assert len(batch["input_ids"][0]) == len(batch["input_ids"][1])
    assert any(not value for row in batch["attention_mask"] for value in row)
    for ids, mask, original in zip(batch["input_ids"], batch["attention_mask"], (candidate, record)):
        assert mask == [value != tokenizer.pad_id for value in ids]
        assert tokenizer.decode(ids) == original


def test_selective_metadata_survives_serialization():
    a, b, c = sp.symbols("a b c")
    state = State(x*(a+b+c), 0)
    actions = enumerate_actions(state, rules=("distribute_term",))
    records = [serialize_candidate(state, action, apply_action(state, action)[0]) for action in actions]
    assert len({json.dumps(record, sort_keys=True) for record in records}) == 3
    assert [action_from_data(record["action"]) for record in records] == list(actions)


def test_model_shape_eval_determinism_and_padding_invariance(tiny, candidate):
    model, tokenizer = tiny
    model.eval()
    batch = tokenizer.batch_encode([candidate, candidate])
    ids = torch.tensor(batch["input_ids"])
    mask = torch.tensor(batch["attention_mask"])
    with torch.inference_mode():
        scores = model(ids, mask)
        assert scores.shape == (2,) and bool(torch.isfinite(scores).all())
        assert torch.equal(scores, model(ids, mask))
        padded_ids = torch.nn.functional.pad(ids, (0, 5))
        padded_mask = torch.nn.functional.pad(mask, (0, 5), value=False)
        torch.testing.assert_close(scores, model(padded_ids, padded_mask), atol=1e-5, rtol=1e-5)


def test_inference_artifact_roundtrip_and_legal_candidates(tiny, tmp_path):
    model, tokenizer = tiny
    state = State(z*(x+y)**2, z*(x**2+2*x*y+y**2))
    before = CandidateScorer(model, tokenizer).score_actions(state)
    directory = tmp_path/"artifact"
    save_artifact(model, tokenizer, directory)
    loaded = CandidateScorer.load(directory)
    after = loaded.score_actions(state)
    assert [item.action for item in before] == [item.action for item in after]
    assert [item.score for item in before] == [item.score for item in after]
    assert all(apply_action(state, item.action)[0] == item.successor for item in after)
    assert loaded.model.training is False
    weights = torch.load(directory/"weights.pt", weights_only=True)
    assert all(isinstance(value, torch.Tensor) for value in weights.values())
    assert set(path.name for path in directory.iterdir()) == {
        "weights.pt", "config.json", "tokenizer.json", "manifest.json", "MODEL.md"}
    assert "Training data is not" in (directory/"MODEL.md").read_text()
    with pytest.raises(ValueError, match="new or empty"):
        save_artifact(model, tokenizer, directory)


def test_inference_rejects_illegal_candidates(tiny):
    model, tokenizer = tiny
    with pytest.raises(ValueError):
        CandidateScorer(model, tokenizer).score_actions(State(x, x), [Action("expand_power")])
    assert CandidateScorer(model, tokenizer).score_actions(State(x, x)) == ()


def test_overlength_and_version_rejection(candidate, tmp_path):
    with pytest.raises(ValueError, match="truncation is disabled"):
        PolynomialTokenizer(10).encode(candidate)
    tokenizer = PolynomialTokenizer()
    with pytest.raises(ValueError, match="format"):
        tokenizer.encode({**candidate, "version": 99})
    with pytest.raises(ValueError):
        tokenizer.decode([2, 1, 7])
    tokenizer.save(tmp_path/"tokenizer.json")
    data = json.loads((tmp_path/"tokenizer.json").read_text())
    data["vocabulary_version"] = 2
    (tmp_path/"tokenizer.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Unsupported tokenizer"):
        PolynomialTokenizer.load(tmp_path/"tokenizer.json")


@pytest.mark.parametrize("data", [["Call", "eval"], ["Q", "1", "0"], ["Q", "02", "1"],
                                  ["P", [["S", "x"], ["Q", "-1", "1"]]],
                                  ["A", [["S", "x"], ["S", "x"]]]])
def test_malformed_or_noncanonical_trees_are_rejected(data):
    with pytest.raises(ValueError):
        expression_from_data(data)


@pytest.mark.parametrize("changes", [{"attention_heads": 3}, {"embedding_dim": 0},
                                     {"dropout": 1}, {"encoder_layers": True}, {"format_version": 2}])
def test_invalid_configuration(changes):
    with pytest.raises(ValueError):
        ModelConfig(**changes)


def test_model_invalid_inputs(tiny):
    model, tokenizer = tiny
    with pytest.raises(ValueError, match="Mask"):
        model(torch.tensor([[2, 0]]), torch.tensor([[True, True]]))
    with pytest.raises(ValueError, match="nonpadding"):
        model(torch.zeros((1, 2), dtype=torch.long), torch.zeros((1, 2), dtype=torch.bool))
    with pytest.raises(ValueError, match="maximum"):
        model(torch.ones((1, 2049), dtype=torch.long), torch.ones((1, 2049), dtype=torch.bool))
    with pytest.raises(ValueError, match="vocabulary"):
        model(torch.tensor([[999]]), torch.tensor([[True]]))
    with pytest.raises(ValueError, match="match"):
        CandidateScorer(model, PolynomialTokenizer(1024))
