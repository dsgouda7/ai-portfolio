"""Code companion for the independent three-objective comparison lab."""

# %% Setup - Run the comparison independently of the main notebook
from pathlib import Path
import sys
import time
import matplotlib.pyplot as plt
import torch

CHAPTER_DIRECTORY = next(
    candidate for candidate in (
        Path.cwd(), Path.cwd() / "learning" / "03-language-model-foundations" / "03-language-models",
    ) if (candidate / "real_text_embeddings.py").is_file()
)
if str(CHAPTER_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(CHAPTER_DIRECTORY))
from language_model import CBOW, SkipGram, EncoderDecoderLM
from encoder_decoder_feedback import responsive_animation_html
torch.set_num_threads(1)
notebook_started = time.perf_counter()

# %% 1 - Load the independent comparison split
from pathlib import Path

import real_text_embeddings
from real_text_embeddings import (
    animate_word_maps,
    blank_shortlist,
    load_real_text,
    plot_losses_and_geometry,
    plot_similarity_heatmaps,
    plot_word_maps,
    project_snapshots,
    report_neighbours,
    shortlist_summary,
    train_comparison,
)
from real_text_embeddings import best_versus_final

real_corpus = load_real_text()

# %% 2 - Record all three training histories
REAL_TEXT_EPOCHS = 16
real_results = train_comparison(
    real_corpus, CBOW, SkipGram, EncoderDecoderLM, epochs=REAL_TEXT_EPOCHS,
)

# %% 3 - Plot recorded loss and geometry
chart_directory = Path(real_text_embeddings.__file__).resolve().parent / "images"
loss_figure = plot_losses_and_geometry(real_results)
loss_figure.savefig(chart_directory / "03a-real-text-loss-and-similarity.png", dpi=130)
plt.show()
plt.close(loss_figure)

# %% 4 - Project all histories through one camera
real_probe_words, real_coordinates, real_pca_fraction = project_snapshots(
    real_corpus, real_results,
)
map_figure = plot_word_maps(real_probe_words, real_coordinates, real_results)
map_figure.savefig(chart_directory / "03a-real-text-word-maps.png", dpi=130)
plt.show()
plt.close(map_figure)

# %% 5 - Compare complete stored rows
similarity_figure = plot_similarity_heatmaps(real_corpus, real_results)
similarity_figure.savefig(chart_directory / "03a-real-text-cosine-heatmaps.png", dpi=130)
plt.show()
plt.close(similarity_figure)

# %% 6 - Play the recorded snapshots
real_animation = animate_word_maps(real_probe_words, real_coordinates, real_results)
real_animation_html = responsive_animation_html(real_animation)
if "get_ipython" in globals():
    from IPython.display import HTML, display

    display(HTML(real_animation_html))
else:
    print(f"Recorded animation rendered ({len(real_animation_html):,} HTML characters).")

# %% 7 - Print actual neighbour changes
report_neighbours(real_corpus, real_results)

# %% 7 - Your turn: Inspect another pair
probe_pair = ("mom", "dad")  # CHANGE THIS: try ("mom", "green"), without retraining
for objective, record in real_results.items():
    pair_ids = [real_corpus["word_ids"][word] for word in probe_pair]
    pair_scores = []
    for rows in (record["snapshots"][0], record["snapshots"][-1]):
        pair_vectors = torch.nn.functional.normalize(rows[pair_ids], dim=1)
        pair_scores.append((pair_vectors[0] @ pair_vectors[1]).item())
    print(f"{objective}: {' / '.join(probe_pair)} cosine "
          f"{pair_scores[0]:+.3f} -> {pair_scores[1]:+.3f}; "
          f"final across-group mean {record['across'][-1]:+.3f}")

# %% 8 - Check held-out answer shortlists
cbow_best = real_results["CBOW"]["best_model"]
for left, right in [("was so", "to have"), ("a little", "named tom"),
                    ("played with", "toys and"), ("they went", "the park")]:
    menu = blank_shortlist(cbow_best, real_corpus, left, right)
    print(f"{left} ___ {right} -> " + ", ".join(f"{word} {chance:.0%}" for word, chance in menu))

shortlist = shortlist_summary(cbow_best, real_corpus)
print(f"\nAll {shortlist['blanks']:,} held-out blanks with a known answer:")
print(f"  right answer is the number-one guess:   {shortlist['first_right']:.1%}")
print(f"  right answer is somewhere in the top 5: {shortlist['shortlist_right']:.1%}")
print(f"  average share given to the first guess: {shortlist['first_share']:.1%}")
print(f"  how alike the 5 shortlisted words' rows are: {shortlist['shortlist_similarity']:+.3f}")
print(f"  how alike 5 random words' rows are:          {shortlist['random_similarity']:+.3f}")

# %% 9 - Check best versus final without retraining
writer_check = best_versus_final(real_results)
writer_record = writer_check["Encoder-decoder"]
if writer_record["final_gap"] < writer_record["best_gap"]:
    print("The writer's group separation faded after its best held-out checkpoint.")
else:
    print("This run did not show fading group separation; judge the recorded losses too.")

# %% 9 - Optional extra-seed robustness check
RUN_SEED_CHECK = False
if RUN_SEED_CHECK:
    for check_seed in (5, 123):
        print(f"--- seed {check_seed}")
        best_versus_final(train_comparison(
            real_corpus, CBOW, SkipGram, EncoderDecoderLM,
            epochs=REAL_TEXT_EPOCHS, seed=check_seed, verbose=False,
        ))
else:
    print("Skipped. Set RUN_SEED_CHECK = True to retrain with seeds 5 and 123.")
print(f"Whole optional lab: {time.perf_counter() - notebook_started:.1f}s "
      "(cache verified, one CPU thread; first download is extra).")
