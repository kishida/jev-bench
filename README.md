# jev-bench

Measuring models that **return a probability for each option instead of writing an answer** — the
Jev / TypeSafe System One style of API, where a request carries a state and a question with named
options and the response carries a distribution over them.

Everything here talks to one endpoint, `POST /v1/systemone`, so it works against any server that
implements it. The reference server is the `jev` branch of llama.cpp
([kishida/llama.cpp](https://github.com/kishida/llama.cpp), see `docs/jev.md` there).

There are two kinds of measurement, and they disagree more than you would expect.

## 1. Classification

1,191 multiple-choice questions with 2 to 8 options, drawn from ARC, MMLU, JCommonsenseQA and BoolQ
and reshaped so the number of options varies. Reports accuracy **and** calibration, because a
probability that does not mean anything is the failure mode this kind of API has to avoid.

The data lives on the Hub as [`kishida/jev-bench`](https://huggingface.co/datasets/kishida/jev-bench),
with the full description of how it was built.

```bash
pip install requests numpy datasets
python eval.py --url http://127.0.0.1:8080
```

`eval.py` needs nothing from this repository — copy it anywhere. It picks its mode automatically: if
the server accepts `options.return_logits`, it fits a temperature on the validation split and
reports accuracy and ECE before and after; if it only returns `probabilities`, it reports accuracy
and whatever calibration the server already applies. The second mode is there for hosted services.

`results/bench_*/systemone_metrics.json` holds the runs behind the table in the llama.cpp docs, and
`results/bench_all.json` collects them. `bench_models.py` is the local driver that starts
llama-server for each GGUF in turn (set `GGUF_DIR`); `eval_systemone.py` is the older evaluator that
still depends on the training project next door.

## 2. Games

Per-step decisions instead of one-shot questions: at every turn the board is sent as a state and one
`choice` question asks which of four directions to take. Nothing is generated, so a game is just a
few hundred classifications.

```bash
python demo/snake/play.py --mode text  --games 10 --url http://127.0.0.1:8080
python demo/snake/play.py --mode image --games 10 --url http://127.0.0.1:8080   # needs --mmproj
python demo/maze/play.py  --mode text --labels --coords --cells 5 --games 5 --url ...
```

`demo/snake` and `demo/maze` both write a full trace to `results/`, replayable in `replay.html`, and
both have a `live.py` that plays forever and streams the board to a browser over SSE.
`demo/extract` is a different shape again: finding the span of a document that matches a condition.

Useful flags: `--minimal` drops the coordinate hints so the text version carries no more information
than the picture, `--board space|table|json|markdown|…` changes how the grid is written, and
`--assistant-prefix` handles models that write something before the label.

## What the results say

- **Classification accuracy does not predict game skill.** Five models inside 0.010 of each other —
  0.914 to 0.923 — average between 0.5 and 4.7 apples at Snake. DeepSeek V4 is the most accurate
  classifier of the three worst players.
- **Check the assistant prefix before judging a model.** A harmony-style model whose label is not
  the first generated token scores like guessing until the prefix is handled — gpt-oss goes from
  0.268 to 0.840.
- **A picture can beat the text.** With the coordinate hints removed so both carry the same
  information, the image version of Snake beats the ASCII one on the models that can see at all.
- **Direction bias is the real failure in the maze.** Aggregate "legal move" rates look respectable
  because a model that walks back and forth in a corridor scores 100% on them. Every model tested so
  far emits only two of the four directions; which two differs per model.

## Layout

```text
eval.py              the classification benchmark, standalone
bench_models.py      run it over a list of local GGUFs
eval_systemone.py    the older evaluator (needs the training project)
demo/snake/          Snake, as a per-step 4-way choice
demo/maze/           the same for mazes
demo/extract/        span extraction
results/             every run, including the traces the viewers replay
jev-bench/           working copy of the Hugging Face dataset repo (not tracked here)
```

## Licence

The code is MIT. The dataset is separate and is `cc-by-sa-4.0`, inherited from the datasets it is
derived from — see the dataset card.
