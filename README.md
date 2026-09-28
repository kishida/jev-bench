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
python demo/snake/play.py --mode text --minimal --games 10 --url http://127.0.0.1:8080
python demo/snake/play.py --mode image --games 10 --url http://127.0.0.1:8080    # needs --mmproj
python demo/maze/play.py  --mode text --labels --coords --cells 5 --games 5 --url ...
python demo/find/play.py  --mode image_axis --bugs 1 --trials 20 --url ...
```

`--minimal` is on in the first line for a reason: without it the state also carries the head and
apple coordinates, and the model can answer without reading the board at all.

[docs/snake.md](docs/snake.md) has the full Snake results and explains why the coordinate-free
condition is the one to compare on.

### Snake

Mean apples over 10 games on a 10x10 board, seeds 0 to 9. Walking into a wall or into its own body
ends the game. The first three columns carry the same information as each other; the fourth adds the
head and apple coordinates, which is why it is only a reference.

| model | text | text, spaced | image | *(+ coordinates)* |
|---|---|---|---|---|
| Qwen3.8 Flash-Next | 1.6 | 2.2 | **6.1** | *6.0* |
| Gemma 4 31B it qat | 2.9 | **4.2** | **4.2** | *5.5* |
| Qwen3.6 35B A3B | 0.0 | 0.0 | **3.7** | *1.4* |
| Qwen3.6 27B | 1.2 | 0.5 | **3.4** | *4.4* |
| Qwen3.8 27B | 0.0 | 0.0 | 0.9 | *4.7* |
| Gemma 4 12B | 0.1 | | 0.3 | *0.8* |
| GLM-5.3-Flash | 0.0 | 0.2 | 0.1 | *1.1* |
| Gemma 4 26B A4B | 0.0 | 0.0 | 0.0 | *0.9* |
| Qwen3-VL 32B | 0.2 | | 0.1 | *0.7* |
| DeepSeek V4 | 0.1 | | 0.0 | *0.5* |
| Qwen3 32B | 0.0 | 0.0 | | *0.0* |
| *greedy, for scale* | | | | *18.5* |

Five of the eleven models never reach an apple without the coordinates. Of those that do, the
picture beats the characters everywhere except Gemma 4 31B, where they tie. Spacing the cells apart
helps the two models that can play at all and does nothing for the rest.

### Maze

Reaching the goal in a maze carved by depth-first search, five mazes per cell size. `solved` is how
many were finished; `legal` is the share of moves that did not walk into a wall. The text version
numbers the rows and columns and states which way each index runs.

| model | 3x3 text | 5x5 text | 3x3 image | 5x5 image | directions used |
|---|---|---|---|---|---|
| Gemma 4 31B it qat | **40%** / 96% | 0% / 92% | 0% / 27% | 0% / 81% | **4** |
| Qwen3.6 27B | **40%** / 85% | 0% / 32% | **40%** / 96% | 0% / 85% | 2 |
| Qwen3.8 27B | **40%** / 85% | 0% / 35% | **40%** / 86% | 0% / 29% | 2 |
| Qwen3.8 Flash-Next | **40%** / 53% | 0% / 34% | 0% / 82% | | 2 |
| Gemma 4 12B | **40%** / 50% | 0% / 81% | 0% / 9% | | 3 |
| Qwen3 32B | 0% / 97% | 0% / 95% | | | 2 |
| Gemma 4 26B A4B | 0% / 76% | 0% / 11% | 0% / 15% | 0% / 9% | 2 |
| Qwen3.6 35B A3B | 0% / 20% | 0% / 32% | 0% / 9% | 0% / 5% | 2 |
| DeepSeek V4 | 0% / 22% | | 0% / 9% | | 2 |
| Qwen3-VL 32B | 0% / 2% | 0% / 7% | 0% / 9% | 0% / 9% | 2 |
| Qwen3 14B | 0% / 5% | 0% / 0% | | | 1 |

Nobody finishes a 5x5. The last column is why the `legal` rate flatters them: **almost every model
emits only two of the four directions**, and which two differs per model. Qwen3 32B scores 97% and
95% on legal moves while solving nothing, because it plays up and down a corridor forever — 352
moves without once choosing left or right. Gemma 4 31B is the only model to use all four, and the
only one to reach a goal from the picture as often as from the text.

### Find the bug

An 8x6 field holds 4 birds, 4 rabbits and 3 bugs, and all 48 cells are the options of one `choice`
question: which cell holds a bug. 20 trials each, as *one bug / three bugs*; guessing scores 2% and
6% respectively. `demo/find/live.py` plays it as a game, where a bug the model finds is taken off the
field and three more are released once they are all caught.

The field is given four ways. The first three are pictures that differ only in what is written on
them; the fourth is characters.

| model | image, axis labels | image, cell names | image, bare | text |
|---|---|---|---|---|
| GLM-5.3-Flash | **100% / 100%** | 100% / 100% | 90% / 90% | 80% / 80% |
| Qwen3.8 27B | **100% / 100%** | 100% / 100% | 85% / 75% | 90% / 80% |
| Gemma 4 12B | **100% / 100%** | 90% / 85% | 65% / 70% | 90% / 95% |
| Qwen3.5 9B | **100% / 90%** | 100% / 90% | 30% / 15% | 30% / 40% |
| Qwen3.6 35B A3B | **100% / 85%** | 100% / 95% | 45% / 55% | 75% / 65% |
| Qwen3.8 Flash-Next | 95% / 100% | 100% / 100% | 65% / 45% | 80% / 90% |
| Gemma 4 31B it qat | 95% / 100% | 100% / 100% | 55% / 55% | 100% / 100% |
| Gemma 4 26B A4B | 90% / 75% | 95% / 100% | 55% / 25% | 100% / 90% |
| Qwen3.5 4B | 75% / 65% | 70% / 50% | 45% / 40% | 20% / 35% |
| Qwen3.5 2B | 60% / 45% | 45% / 40% | 5% / 10% | 5% / 10% |
| Qwen3.6 27B | | 100% / 100% | 55% / 45% | 80% / 85% |

**A bare picture measures the wrong thing.** With nothing written on it, the prompt has to say that
the columns run A to H and the rows 1 to 6, and the model has to hold that against what it sees.
Printing the letters and numbers in the margin, the way a spreadsheet does, is worth 10 to 70 points
— Qwen3.5 9B goes from 30% to 100% — and what remains is the reading of the grid itself.

Printing a name inside every cell is not better than labelling the margins, and for the two smallest
models it is worse, presumably because the text competes with the animals for the picture.

Unlike Snake and the maze, this one rises with size in an orderly way, and saturates: six of the
eleven models are at or above 95%. Only the bare picture still separates them.

`demo/snake`, `demo/maze` and `demo/find` write a full trace to `results/`; the first two replay it
in `replay.html`, and all three have a `live.py` that runs forever and streams the board to a
browser over SSE. `demo/extract` is a different shape again: finding the span of a document that
matches a condition.

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
- **Hints change what is being measured.** Telling the model where the head and the apple are lets
  it subtract two coordinates instead of reading the board, and Qwen3.8 27B goes from 0 apples to
  4.7. Compare board formats with the hints off, or the comparison measures nothing.
- **A picture beats the text once the hints are off.** Five of six models score highest on the image.
  Qwen3.6 35B A3B manages 0.0 from characters and 3.7 from a picture of the same board.
- **Write the coordinate system on the picture.** Telling the model in words that the columns run A
  to H costs it more than reading the grid does: printing the letters and numbers in the margin
  takes Qwen3.5 9B from 30% to 100% at finding a bug.
- **Direction bias is the real failure in the maze.** Aggregate "legal move" rates look respectable
  because a model pacing up and down a corridor scores 100% on them. Almost every model emits only
  two of the four directions — Qwen3 32B played 352 moves without once choosing left or right — and
  which two differs per model. Gemma 4 31B is the first to use all four, and the only one to reach
  the goal.

## Layout

```text
docs/snake.md        what the Snake numbers mean, and how the conditions differ
eval.py              the classification benchmark, standalone
bench_models.py      run it over a list of local GGUFs
eval_systemone.py    the older evaluator (needs the training project)
demo/snake/          Snake, as a per-step 4-way choice
demo/maze/           the same for mazes
demo/find/           find the bug in a field of 48 cells
demo/extract/        span extraction
results/             every run, including the traces the viewers replay
jev-bench/           working copy of the Hugging Face dataset repo (not tracked here)
```

## Licence

The code is MIT. The dataset is separate and is `cc-by-sa-4.0`, inherited from the datasets it is
derived from — see the dataset card.
