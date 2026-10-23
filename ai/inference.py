"""Inference-only artifacts and verified legal-candidate scoring."""

from dataclasses import dataclass
import json
from pathlib import Path

import torch

from mathai import Action, State, apply_action, enumerate_actions
from .model import ActionTransformer, ModelConfig, PolynomialTokenizer, serialize_candidate

MODEL_CARD = """# MathAI candidate scorer

Small Transformer encoder producing relative candidate-action scores for exact
polynomial proof tasks. Legal moves and correctness come from the symbolic
environment, not this model. Scores are not correctness probabilities and
ranking accuracy does not establish search improvement. Training data is not
released. This artifact contains inference components only. No license for
weights or datasets is selected by this export. Quality depends on training;
smoke-test weights are not a useful trained proof policy.
"""


def save_artifact(model, tokenizer, directory):
    if (model.config.vocab_size != tokenizer.vocab_size
            or model.config.max_sequence_length != tokenizer.max_length):
        raise ValueError("Model and tokenizer configurations do not match")
    directory = Path(directory)
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("Artifact directory must be new or empty")
    directory.mkdir(parents=True, exist_ok=True)
    torch.save({key: value.detach().cpu() for key, value in model.state_dict().items()}, directory/"weights.pt")
    model.config.save(directory/"config.json")
    tokenizer.save(directory/"tokenizer.json")
    (directory/"manifest.json").write_text(json.dumps({"artifact_version": 1, "serialization_version": 1}), encoding="utf-8")
    (directory/"MODEL.md").write_text(MODEL_CARD, encoding="utf-8")


@dataclass(frozen=True)
class ScoredCandidate:
    action: Action
    successor: State
    score: float


class CandidateScorer:
    def __init__(self, model, tokenizer, device="cpu"):
        if (model.config.vocab_size != tokenizer.vocab_size
                or model.config.max_sequence_length != tokenizer.max_length):
            raise ValueError("Model/tokenizer mismatch")
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.tokenizer = tokenizer

    @classmethod
    def load(cls, directory, device="cpu"):
        directory = Path(directory)
        manifest = json.loads((directory/"manifest.json").read_text(encoding="utf-8"))
        if manifest != {"artifact_version": 1, "serialization_version": 1}:
            raise ValueError("Unsupported inference artifact")
        config = ModelConfig.load(directory/"config.json")
        tokenizer = PolynomialTokenizer.load(directory/"tokenizer.json")
        model = ActionTransformer(config)
        model.load_state_dict(torch.load(directory/"weights.pt", map_location="cpu", weights_only=True), strict=True)
        return cls(model, tokenizer, device)

    def score_records(self, records, batch_size=16):
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        records = list(records)
        scores = []
        self.model.eval()
        with torch.inference_mode():
            for offset in range(0, len(records), batch_size):
                batch = self.tokenizer.batch_encode(records[offset:offset+batch_size])
                tensors = {key: torch.tensor(value, device=self.device,
                           dtype=torch.long if key == "input_ids" else torch.bool) for key, value in batch.items()}
                scores.extend(self.model(**tensors).cpu().tolist())
        return tuple(scores)

    def score_actions(self, state, actions=None, *, rules=None, batch_size=16):
        actions = tuple(enumerate_actions(state, rules=rules) if actions is None else actions)
        successors = [apply_action(state, action)[0] for action in actions]
        records = [serialize_candidate(state, action, successor) for action, successor in zip(actions, successors)]
        scores = self.score_records(records, batch_size)
        return tuple(ScoredCandidate(action, successor, score)
                     for action, successor, score in zip(actions, successors, scores))
