# MathAI candidate-action scorer

The public `ai` package implements a small Transformer encoder and inference
loading. Training tools, generators, datasets, notebooks, checkpoints and logs
are maintained separately and are not released with this repository. No trained
weights are included. No dataset or weight license has been selected.

The default configuration has four encoder layers, four attention heads,
embedding dimension 128, feed-forward dimension 512, dropout 0.1 and a maximum
sequence length of 2048. It has **1,089,409 parameters**. `ModelConfig` validates
and saves configurable dimensions. Learned token/position embeddings feed a
bidirectional encoder, a padding-masked mean pool, LayerNorm and a scalar head.
Encoder layers are independently initialized. Higher scores indicate model
preference, not mathematical correctness or calibrated probabilities.

## Serialization and tokenization

Serialization version 1 uses JSON trees: `S` for symbols, `Q` for exact rational
numerator/denominator strings, `A`/`M` for canonical sums/products, and `P` for
powers. Child order follows the environment's canonical SymPy structure. There
is no polynomial expansion for serialization, so equivalent factored and
expanded expressions remain distinct. Decoding uses allowlisted constructors
and rejects noncanonical/unsupported trees; it never evaluates source text.

Each candidate contains `[CURRENT]`, `[TARGET]`, `[ACTION]`, and `[NEXT]` fields.
The action includes the rule, integer path, integer selection tuple and expected
subexpression. `serialize_candidate` verifies the supplied transition through
the existing environment. Tokenization validates format and structural data;
it does not itself certify that arbitrary supplied records are legal moves.

Vocabulary version 1 is fixed: IDs 0–7 represent `[PAD]`, `[UNK]`, `[BOS]`,
`[CURRENT]`, `[TARGET]`, `[ACTION]`, `[NEXT]`, `[EOS]`; IDs 8–263 encode UTF-8
bytes 0–255. Payload bytes cannot masquerade as field-separator IDs. All symbol
names, signs, integer digits and exact rational coefficients are lossless;
`[UNK]` is reserved and not emitted for valid UTF-8 text. No vocabulary is fitted
to training, validation or test examples. Tokenizer files record every byte ID,
special token, both versions, maximum length and the overlength policy.

Attention masks are boolean, true for real tokens and false for padding.
Overlength inputs raise an error; no truncation is permitted. Callers must
explicitly handle tasks outside the configured context window. Expressions
still obey SymPy's documented automatic evaluation conventions.

## Inference

Install the root requirements with an ordinary supported CPython/PyTorch build.
The existing symbolic environment and BFS remain independent of `ai` and do
not import PyTorch or use learned scores.

```python
from ai.inference import CandidateScorer
from mathai import State, parse_expression

scorer = CandidateScorer.load("path/to/inference-artifact", device="cpu")
state = State(parse_expression("x**2-y**2"), parse_expression("(x-y)*(x+y)"))
candidates = scorer.score_actions(state)  # Legal, verified actions in environment order.
ranked = sorted(candidates, key=lambda item: -item.score)  # Stable ties.
```

Inference artifacts contain only `weights.pt`, `config.json`, `tokenizer.json`,
`manifest.json`, and `MODEL.md`. Loading uses strict state-dict validation and
PyTorch's `weights_only=True`; configuration/serialization versions and model
versus tokenizer lengths must match. Artifact saving refuses a nonempty output
directory. Load artifacts only from sources you trust; this is not a general
untrusted-file sandbox. See the official [TransformerEncoder documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.TransformerEncoder.html)
and [serialization API](https://docs.pytorch.org/docs/stable/generated/torch.load.html).

The training objective is grouped ranking with probability mass assigned to
known successful continuations, allowing multiple positives. Other legal actions
may also be useful. Validation top-1 known-positive accuracy and MRR therefore
measure partial-label ranking, not completeness or search success. Dataset
splits and generation audits are private. Training data is not released.

Only tiny CPU smoke training is validated locally. Full training, CUDA/mixed
precision, and Colab execution require separate execution. No useful trained
policy or measured search improvement is claimed. End-to-end model-guided
search evaluation remains pending; BFS is the unchanged baseline.
