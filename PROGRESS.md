# Progress

Live task state. Updated at the end of every task. A fresh session reads this to resume.

**Phase 1 — data and evaluation infrastructure. No generative model yet.**

## Next action

> **Phase 1 is complete (13/13). Phase 2 has begun and the model generates.**
>
> On 2026-08-31 the project produced handwriting for the first time: held-out CVL
> writer 0057, one real line as style, three lines that writer never wrote, in a
> visibly matching hand. See `docs/phase2-first-generation.md` and
> `outputs/probe_lines/lines_contrast.png`.
>
> ### The references, measured on lines and on English only
>
> T15 did not need Colab -- `check_metrics.py` generates nothing, it only runs
> Inception, the embedding and TrOCR over real images. Measured on CPU and
> committed to `references/references_cvl_lines_64.json`:
>
> | | on **lines**, English only | on words (phase 1) |
> |---|---|---|
> | FID floor | **19.15** | 33.72 |
> | writer top-1 | **85.8%** | 66.9% |
> | writer top-5 | **94.4%** | 90.1% |
> | CER on real lines | **11.45%** (300 lines) | 12.33% (40 lines, one writer, unfiltered) |
>
> **The FID floor nearly halved against the word-level one.** A line holds a
> whole sentence, so lines vary less from one another in Inception's feature
> space than isolated words do. A generated set scoring 60 would have read as
> 1.8x the floor against words and is in fact 3.1x against lines. The word-level
> number would have flattered every result this project is about to produce.
>
> **Writer retrieval is much higher than it looked.** More handwriting per image
> means more evidence of the hand, so the bar the generator has to clear rose
> from 66.9% to 85.8%.
>
> **Excluding the German passage took 2.28 points off CER**, from 13.73% to
> 11.45%. That difference is the size of the penalty we were imposing on TrOCR
> for reading a language it was never trained on. 11.45% is its honest error rate
> on English handwriting.
>
> The two retrieval figures come from different samples (126 writers against 116)
> and are not directly comparable with the German-inclusive run. CER is.
>
> **CER cannot be measured on Colab.** It reads the *raw* CVL line images rather
> than the pack, and the 5 GB of sources are deliberately not copied to the VM.
> That is why `references.update` merges rather than replaces: a machine that
> measures two of the three must not delete the third.
>
> ### T16 ran. The project has numbers.
>
> 2026-09-09, T4, 53 minutes, 300 lines in the hands of 94 unseen writers.
> Full write-up in `docs/phase2-first-evaluation.md`.
>
> | | generated | real | no style at all |
> |---|---|---|---|
> | FID | **63.92** | 19.15 | 254.29 |
> | writer top-1 | **20.5%** | 85.8% | 1.7% |
> | writer top-5 | **43.6%** | 94.4% | — |
> | CER | **33.14%** | 12.06% | — |
>
> **It produces handwriting** — 81% of the way from printed text to real, on the
> FID scale. **The identity carries only partly** — retrieval is 18x chance, so
> the style input is genuinely used, but only 23% of the way from chance to what
> real handwriting scores. Emuru writes convincing handwriting that is only
> partly the right person's.
>
> The run completed where the previous one died: at request 41 the model declined
> to write, three re-draws failed, and it was excluded with its ground truth
> rather than taking the run down. Four others were saved by a retry.
>
> ### The first trustworthy style figure -- Emuru, T23's run
>
> `notebooks/colab_eval.ipynb` cells 1 to 6 on a T4: 300 requests, 295 kept, 53.9
> minutes of generation (0.09 lines/s, the rate of 2026-09-09).
>
> | | generated | real | typeface |
> |---|---|---|---|
> | **HWD** | **2.00 [1.88, 2.13]** | 0.86 [0.81, 0.91] | 2.99 [2.91, 3.06] |
>
> **Emuru's output sits 53% of the way from real handwriting to a typeface**, and
> its interval is nowhere near the real one. Reference: 1,104 real lines over 93
> writers, none a target or a style line.
>
> The rest: FID 66.72 [60.56, 72.88] · writer top-1 20.3% [15.9, 25.1] · CER 28.1%
> [23.9, 32.4] against 11.3% real, gap +16.8 · 13 of 295 truncated (4.4%) · 5
> excluded as empty, 1 saved by a retry. The earlier runs' FID (63.92, 69.46) and
> retrieval (20.5%, 22.1%) all fall inside this run's intervals. Truncation halved
> against the 8.7% of 2026-09-10 on the same requests; not explained.
>
> The harness check before it (cell 5, fake generator, 120 samples) printed
> generated = typeface = 3.02 at 100% and real 1.10 -- HWD's first run on a GPU.
>
> **What 53% does not say.** The typeface is the far end for *no handwriting at
> all*, not for *someone else's handwriting*. HWD 2.00 mixes two distances -- not
> looking like real handwriting, and not looking like this writer -- and the
> typeface anchor cannot separate them. Retrieval had a chance level for exactly
> this; HWD does not have one yet.
>
> ### Emuru carries more than half of a writer's identity -- T24's run, 2026-09-14
>
> Cell 6 again, same 300 requests, now with the identity block. 296 kept, 56.5
> minutes of generation, saved to Drive (`results/eval_emuru_lines`, 296 images).
>
> | | gap to the other 92 writers | own writer nearest |
> |---|---|---|
> | generated | 1.05 [0.94, 1.15] | 47.3% [37.6, 57.0] |
> | real | 1.85 [1.77, 1.94] | 97.8% [94.6, 100.0] |
> | typeface | -0.01 [-0.08, 0.07] | 1.1% [0.0, 3.2] -- chance 1.1% |
>
> **Identity: the generated lines carry 56.5% [50.9, 61.9] of what real lines do.**
> The typeface sits at zero as it must, so shared text does not create the gap.
> Distance repeated the first run: HWD 2.03 [1.90, 2.16] against 0.86 real and
> 2.99 typeface, 55% of the way. FID 67.70 [61.81, 73.59], retrieval 20.9%
> [16.6, 25.7], CER 30.4% against 11.3% (gap +19.1), 20 of 296 truncated (6.8%),
> 4 excluded.
>
> **This changes the reading of the project's central number.** Retrieval put
> Emuru a quarter of the way from chance to real; identity puts it past half.
> The gap is what T21 predicted -- retrieval collapses under blur and HWD does
> not. Nearest-writer (47.3%, per writer over their 1-7 lines) and retrieval
> (20.9%, per single line) are not the same question and are not compared.
>
> Two cautions. "Identity" is whatever sets one CVL writer's lines apart from
> another's, and that includes their pen -- which a real user's photographs would
> share too, but which is not the hand alone. And each run builds its own
> reference, so 7b's figure is compared through its own real anchor, not raw.
>
> ### Two style lines side by side make Emuru worse -- T27, 2026-09-14
>
> Cell 7b: the same 300 requests with `--style-refs 2`, the two lines joined into
> one wide image and their texts joined with a space (`nib.models.style.join_style`).
> 295 kept, 86.4 minutes of generation against 56.5, saved to Drive
> (`results/eval_emuru_lines_refs2`).
>
> | | one line (cell 6) | two lines (7b) | |
> |---|---|---|---|
> | HWD identity | 56.5% [50.9, 61.9] | **42.3% [36.2, 48.7]** | separate |
> | own writer nearest | 47.3% [37.6, 57.0] | 25.0% [16.3, 33.7] | separate |
> | HWD distance | 2.03 [1.90, 2.16] | 2.22 [2.07, 2.36] | overlap |
> | FID | 67.70 [61.81, 73.59] | 90.64 [80.97, 100.30] | separate |
> | CER | 30.4% [25.7, 35.2] | **59.6% [54.6, 64.4]** | separate |
> | writer top-1 | 20.9% [16.6, 25.7] | 18.6% [14.2, 23.1] | overlap |
> | truncated / excluded | 6.8% / 4 | 8.1% / 5 | |
>
> The criterion was fixed before the run -- clear 61.9% -- and it fell the other
> way: worse on identity, on FID, and CER nearly doubled. Real lines scored the
> same in both runs (0.86, identity gap 1.85 against 1.81), so the ruler did not
> move; the output did.
>
> **Closed for Emuru: more evidence as a wider image.** Not closed: more evidence.
> The collapse in CER says the model is not writing the target text, which points
> at how two lines reach it rather than at the idea. Unverified hypothesis: Emuru
> puts the style text and the target text through T5 together, and with two
> lines' worth of prefix it loses where the style ends -- writing style words, or
> skipping target words. The saved samples decide it without a GPU.
>
> ### What the samples show -- no GPU, 2026-09-14
>
> Both runs downloaded from Drive to `outputs/`. Each run's requests were rebuilt
> from the seed (296/296 and 295/295 matched the saved keys), so every sample's
> style text is known. Scratch script, sheets of samples looked at by eye.
>
> **Emuru's output is bimodal, not uniformly mediocre.** One style line: median
> per-sample CER **9.3%** against a mean of 30.8%. The mean is carried by a tail --
> 10% of lines above 90% CER -- and those lines are not bad handwriting, they are
> no handwriting: two or three words, then a smear of repeated vertical strokes to
> the end of the budget, or a blank. The median line reads well and looks like a
> hand.
>
> **Two lines break the text, not only the tail.** Median CER 54.4%, 21% above
> 90%. Typical two-line outputs repeat words ("has has", "trong trong"), drop
> words, and pull words in from the style text ("plant" from "plants and
> animals"). Worst cases are the same smear as above. CER rises with the combined
> style text: 33% under 60 characters, 79% over 100. The hypothesis that the model
> loses the boundary between style and target text fits what is on the page. Per
> writer, two lines beat one for 31% of the 91 writers in both runs.
>
> **The style line matters even at one.** Style lines under 500px: mean CER 69%,
> failures 22% (n 18). 500-1100px: 26-28%, failures 9-10% (n 242). Over 1100px:
> 37% (n 36). A page offers a choice of style line; the evaluation drew one at
> random.
>
> **Catching broken lines would raise identity too, not only CER.** Recomputed from
> the saved images and reference keys -- which reproduced cell 6 exactly, 56.5% /
> 47.3% / 30.4% -- with unreadable samples removed from generated and real alike:
>
> | kept | samples | writers | identity | own writer nearest | CER |
> |---|---|---|---|---|---|
> | all | 296 | 93 | 56.5% | 47.3% | 30.4% |
> | CER <= 90% | 267 | 91 | 60.0% | 47.3% | 21.1% |
> | CER <= 50% | 224 | 86 | 67.3% | 52.3% | 9.5% |
>
> A bound, not a measurement of a fix: removal is not a re-draw, it can remove
> hard writers along with bad draws (93 writers fall to 86), and these carry no
> intervals. It says the broken tail costs identity as well as legibility.
>
> **Fine-tuning is available.** The Emuru repository
> (`github.com/aimagelab/Emuru-autoregressive-text-img`, MIT) ships training code,
> per `docs/research-2026-08-28-vatr-line.md`; reported on one 4090. Whether it
> adapts to one writer from 22 lines without overfitting is untested.
>
> ### The goal, restated by Amri -- 2026-09-14
>
> **A system that works**: photograph one or two pages of your own handwriting,
> type any text, get it back in your hand. Not a portfolio piece and not a paper.
> New approaches are welcome; the measure is whether it works in the end. GPU is
> not a constraint -- Amri will buy more Colab units when needed.
>
> Two things follow. The product will be given a **page**, 20 to 30 lines, while
> every evaluation so far gave the model **one line** -- so the number of style
> lines is the main experimental axis, not a detail. And the VM that produced HWD
> 2.00 was reclaimed before cell 7 ran: the images are gone and only the printed
> figures above survive.
>
> ### The plan
>
> 1. **Know where we stand.** T24's identity anchor, then one GPU session:
>    notebook cells 1-6 (Emuru, one style line) and 7b (two lines). Every run
>    cell now saves to Drive itself. Pinning the Emuru and Eruku Hugging Face
>    revisions first would keep every experiment on the same model -- the Colab
>    run warned their remote code was downloaded fresh.
> 2. **Amri's own handwriting.** First pipeline component: split a photographed
>    page into lines. Then a probe -- his lines as style, new sentences -- and a
>    small blind test.
> 3. **Fidelity, one experiment at a time, cheapest first**, each closed if the
>    identity figure does not move outside its interval: pick the most
>    representative style line; more style lines (side by side made Emuru worse at two, T27;
>    Eruku takes any width and is untested at two or more); best-of-N against
>    the writer's other lines; per-writer fine-tuning, **which needs Amri to
>    reopen "no per-user training"**; another model only if those fail.
> 4. **Finish the pipeline**: transcribe the style lines or use a dictated
>    passage, generate line by line, lay out a page, output, interface.
>
> Deprioritised: retraining the writer-retrieval embedding. The identity anchor
> and a blind test cover what it was for.
>
> **Awaiting Amri:** (a) per-writer fine-tuning as an experiment, yes or no;
> (b) the definition of "works" -- proposed: people who know his handwriting
> cannot pick his real line out of a real/generated pair much better than 50%.
>
> ### The immediate next task
>
> **Recommended to Amri, awaiting his choice: generation with quality control.**
> Use the page by *selection*, not concatenation. For each line to write: pick
> style lines from the page in the length range that works (500-1100px), draw
> several candidates, read each with TrOCR against the intended text, reject the
> broken ones, keep the best. Selecting by readability first, because it is
> independent of HWD; selecting by closeness to the writer comes second and must
> use page lines disjoint from the ones identity is measured against, or it
> measures itself. Needed in the product whatever else is done, fine-tuning
> included. Costs candidates x generation time: 11 s a line measured, so four
> candidates is about 45 s a line before any batching.
>
> **Next after it, needs Amri's yes: per-writer fine-tuning** on page 1 of the
> passage, measured on page 2. The largest untested lever on identity; training
> code exists.
>
> Closed: joining style lines side by side for Emuru. Open but lower: Eruku with
> several lines (untested, 128 min a run).
>
> `analyse_run.py` does not read HWD yet; cell 9 needs that before it can compare
> two runs on style. Cells that should not be run sit below a "kept for
> reference" divider, commented out.
>
> **In parallel, Amri:** copy out `configs/passages/english_page1.txt` and
> `english_page2.txt` by hand, one sheet each, keeping every line break exactly, in
> pen on plain paper with margins, without crossing out. Photograph each from above
> in daylight and copy the original files -- not through WhatsApp, which
> recompresses -- to `data/raw/personal/` as `passage_page1` and `passage_page2`.
>
> **Then, Claude:** split those photographs into lines (22 per page expected), pair
> each line with its passage text, and run the first probe on Amri's own hand:
> page 1 as style, page 2's lines generated and set beside his real page 2.
>
> ### HWD's floor was not a floor -- 2026-09-11, T23
>
> HWD is the distance between each writer's *mean* feature on either side, and a
> mean over few lines is noisy -- noise that adds to the distance even when both
> sides are the same person's real hand. Real lines against other real lines by
> the same 90 held-out writers, k lines per writer per side:
>
> | k | 1 | 2 | 3 | 6 | 10 |
> |---|---|---|---|---|---|
> | HWD, real vs real | 1.760 | 1.255 | 1.005 | 0.704 | 0.539 |
>
> Under the exact sampling of a 300-sample run (93 writers, 1 to 7 lines each),
> real against real read **1.055 / 1.063 / 1.107** over three seeds. The 0.641
> quoted as the floor came from an uncommitted script at about seven or eight
> lines per writer, judging by the curve. Cell 6 as it stood would have reported
> even a perfect generator as nearly a fifth of the way to no hand at all.
>
> Content barely registers: a typeface scored 3.080 against the target's own text
> and 3.075 against a different one.
>
> **What changed.** Every run now scores three aligned sets against one shared
> reference -- up to 12 real lines per writer that are neither a target nor a
> style line: the generated lines, the real target lines (same writers, texts and
> counts), and the same texts in a typeface. Generated and real differ in nothing
> but being generated. HWD carries a 95% interval by resampling writers, its
> per-writer terms go into `analysis.npz`, and the per-writer computation matches
> the package's own `HWDScore` to 2e-8 on real lines. An HWD failure no longer
> takes CER or the saved bundle with it.
>
> Also found on the way. Cell 2 imported `nib` inside the Colab kernel, where
> `/content/nib` -- the clone -- reads as an empty namespace package, and failed
> with `No module named 'nib.engine'`; it now checks in a fresh interpreter, which
> is where the scripts run. And `hwd` has never been importable locally: its
> `editdistance` dependency has no Windows wheel for Python 3.13, which is why
> 0.641 was measured through a workaround nobody saved.
>
> ### The task before it, for reference
>
> **Re-run T16 with the corrected token budget.** 10.7% of the output was
> truncated, and a line cut short loses its ending to deletion errors, so part of
> the CER gap is ours rather than the model's. `TOKENS_PER_CHAR` moved from 4.0
> to 5.5 (above the 99th percentile of 5.04, measured over 1,524 lines) and
> `MAX_TOKENS` from 256 to 384, since at 5.5 a cap of 256 would clamp every line
> past 47 characters. The share of real lines that do not fit their budget falls
> from 5.9% to 0.7%.
>
> Nothing else changes: same pack, same references, same notebook. Re-run the
> install cell to pull the fix, skip cell 5, run cell 6.
>
> ### Setting a run up from scratch, for reference
>
> 1. Upload `data/processed/upload/cvl_lines_64.lmdb` (**127 MB, 9,142 records**)
>    to `MyDrive/nib/`. **Never** the copy in `data/processed/` -- LMDB reserves
>    its map size up front, so that one is 8 GB on the wire. The two packs this
>    project has built share a filename and differ by 1,720 records, so check the
>    size: 148 MB is the older one, which still contained the German passage.
> 2. Open `notebooks/colab_eval.ipynb` and run it top to bottom.
>
> It also needs `MyDrive/nib/checkpoints/writer_embedder.pt`, which should
> already be there from the T11 run.
>
> `evaluate_generator.py` refuses to start when the pack and the references
> disagree on record count, so uploading the wrong one costs seconds rather than
> an hour. Skip cell 5 -- the raw images it needs are not on the VM.
>
> Watch the truncation count in the output. It is the check on whether the new
> token budget is right; if it is high, `TOKENS_PER_CHAR` in
> `src/nib/models/emuru.py` is too low.
>
> Phase 2 runs T13 -> T16. Only T16 is left, and only it needs a GPU.
>
> ### Why the plan grew from two steps to four (2026-08-31)
>
> Measuring the line data before writing the pack turned up three things. All
> three were invisible while the working unit was a word, and all three become
> load-bearing the moment it is a line.
>
> **1,157 lines carry a transcription that is missing a word.** CVL dropped word
> crops whose segmentation failed, but the line image still holds that word's
> ink. Pairing the two charges the recogniser a deletion error for reading
> correctly. Confirmed by counting ink blobs: gap lines carry 0.58 more per line
> than complete ones, relative to their own word count. They are dropped and
> counted. 9,142 clean lines remain after the German passage comes out too.
>
> **The token budget was set for words and never revisited.** Emuru's VAE
> compresses width by 8, so one token is 8px (`lengths / 8` in its
> `modeling_emuru.py`). `EmuruGenerator` passes `max_new_tokens=96` = 768px. A
> real CVL line at 64px averages **886px**, p90 1198px. *It could not finish an
> average line.* The two 756px outputs recorded as "runaway generation" were
> 94-95 tokens: they hit our cap. The model's own default is 256. T16 replaces
> the fixed cap with a per-request budget from the text length, and counts every
> truncation.
>
> **The three reference numbers were measured on words.** FID 33.72 and
> retrieval 66.9% both came from `cvl_words_64.lmdb`; CER 12.33% came from 40
> unfiltered lines of a single writer. Scoring generated *lines* against a
> word-level FID floor compares different things. T15 re-measures all three on
> the line pack -- one run of `check_metrics.py --pack ...`, no new code.
>
> A fourth item from `PROGRESS.md` turned out to need no work: it asked for
> `normalise_ink` on the line path, but `normalise_word` already called it
> internally. Raw lines sit at 2nd-percentile brightness ~130; after
> normalisation ~21. The faint probe output came from loading raw images.
>
> ### Product idea, Amri, 2026-09-10: enrol from a dictated passage
>
> Instead of asking a new user to photograph a page they already have, hand them
> a printed passage and ask them to copy it. The transcription is then known
> exactly, because we chose it -- no recogniser in the loop and no error
> inherited from one.
>
> This is how CVL itself was built, which is why 99,904 of its words carry a
> transcription nobody typed.
>
> Two things it buys beyond the obvious. The passage can be **designed**: one
> that contains every character in the charset guarantees the system has seen
> how this person forms a `Q` and a semicolon, rather than hoping they turned
> up. And it stays worth doing even with Eruku, whose `style_text` is optional
> but documented as helping -- a dictated passage gives the best possible one for
> free rather than merely avoiding the worst.
>
> The cost is that it is a different product. "Copy this passage" works for
> enrolment and does not work on a page that already exists -- an old letter, a
> diary, a grandparent's hand. Both may be wanted; they are not the same feature.
>
> ### The style metric was broken, and is now replaced -- 2026-09-11
>
> **Read this before trusting any retrieval figure in this file.** A *real* line,
> by unquestionably the right writer, blurred by 0.8 pixels:
>
> | | ours (retrieval) | HWD |
> |---|---|---|
> | other real lines | 96.8% | 0.641 |
> | the same lines, blurred 0.8px | **12.2%** | 0.721 |
> | the same lines, resampled 1/2 | 10.1% | 0.636 |
> | the same text in a typeface | -- | 2.931 |
>
> Blur does not change whose handwriting something is, and every generative
> decoder produces exactly that softness. So writer retrieval has been reporting
> sharpness as much as style, and **Emuru's 22.1% is better than a real line by
> the correct writer put through a mild blur.** The comparisons of 2026-09-10 --
> Emuru against Eruku, the cfg sweep -- are all confounded by output sharpness.
>
> `nib.engine.metrics.hwd` replaces it as the primary style measure. HWD is a
> VGG16 trained on 100M rendered lines, it moves 12% under the damage that costs
> ours 87%, it separates handwriting from a typeface by 4.5x, and it is what
> Emuru's and Eruku's own papers report -- so our figures become comparable to
> published ones for the first time. Optional extra: `pip install -e ".[hwd]"`.
>
> Retrieval stays, because a percentage is far more legible than a distance and
> two agreeing measures beat one -- but it needs retraining with blur, resampling
> and noise in its augmentation before its number means what it says. About 20
> minutes on a T4, per the T11 notes.
>
> **Next run:** `--generator emuru --samples 300` with the hwd extra installed.
> That is the project's first honest style number.
>
> **The HWD column above is relative, not absolute.** Its figures came from a
> script that was never committed, at a lines-per-writer nobody recorded. Within
> that one protocol the reading holds: blur moved HWD 12% and a typeface sat 4.5x
> away. As a floor for a run, 0.641 does not; see T23 above.
>
> ### Where the project stood before that, 2026-09-10
>
> | | Emuru | Eruku cfg 1.25 | Eruku cfg 1.0 | real |
> |---|---|---|---|---|
> | FID | **69.5** | 80.7 [73.2, 88.3] | -- | 19.15 |
> | writer top-1 | **22.1%** | 5.3% [3.0, 8.0] | 10.0% [3.3, 18.3] | 85.8% |
> | CER gap | +20.4 | **+13.0** | -- | -- |
> | empty outputs | 6 | **0** | **0** | -- |
>
> Eruku writes the text better and the hand worse. Its learned end-of-generation
> token does fix one failure completely -- Emuru needed four retries and lost two
> requests outright, Eruku lost none -- and does nothing for truncation, which
> sits at 8% for both.
>
> `cfg_scale` is the dial: dropping it from 1.25 to 1.0 doubled retrieval. The
> intervals at 60 samples overlap, so this is a direction and not yet a finding;
> cfg 2.0 completes the line.
>
> **The number that has not moved is the project's whole claim.** Writer
> retrieval is 22.1% at best against a ceiling of 85.8% -- a quarter of the way
> from chance to what real handwriting scores. Everything else is now measured
> carefully; this is not solved.
>
> The largest untried lever on it: **more than one line of style**. The model is
> given a single line and asked to learn a hand. Emuru's interface takes one
> image, but a wider image holding several concatenated lines is still one image,
> and Eruku takes any width.
>
> ### Known and open
>
> - `find_page` returns the whole frame for `data/raw/personal/dim.jpeg`, so the
>   black cloth above the sheet reaches line segmentation as six false lines.
> - Line segmentation loses lightly written strokes: the first "The" on Amri's
>   sample has 63 pixels below the ink threshold. Hysteresis recovers it and brings
>   the grid back with it (16 lines on the WhatsApp copy), so it is built and off.
> - `data/raw/personal/page1_transcription.txt` is split by sentence, not by written
>   line, and differs from the page ("wanderd", "Project"). The dictated passage
>   replaces it.
> - Emuru needs the style sample's *transcription*. A user photographing a page
>   has transcribed nothing, so the product must read it first -- TrOCR is already
>   here for that, at TrOCR's accuracy. This constrains the architecture.
> - Emuru also ships `generate_batch`, which `EmuruGenerator` does not use. It
>   takes per-sample `lengths` and would cut Colab generation time. Not needed
>   for correctness; worth doing if T16 is slow.
> - **Run evaluation on Colab, not locally.** 220s per line on CPU; minutes on a T4.
> - `data/processed/cvl_words_64.lmdb` shows 8 GB but holds 469 MB; it is sparse
>   and locked by a stale handle. The compacted copy for uploading is at
>   `data/processed/upload/cvl_words_64.lmdb`. A reboot clears the lock.
> - transformers is pinned to 4.x. 5.x cannot load Emuru and breaks TrOCR's
>   tokenizer -- the same version gap causes both.

## Status

| ID | Task | Status | Verified by |
|----|------|--------|-------------|
| T0 | Repo skeleton, packaging, lint, git init | **done** | ruff clean · ruff format clean · pytest 2 passed |
| T1 | Config system (typed schema, single path root) | **done** | ruff clean · 16 tests passing |
| T2 | charset — char/index mapping | **done** | ruff clean · 31 tests passing |
| T3 | Synthetic IAM fixture generator | **done** | ruff clean · 49 tests · 20 writers / 1000 words in 5.7s |
| T4 | IAM parser | **done** | ruff clean · 72 tests (5 skipped, awaiting real IAM) |
| T5 | Writer-disjoint split | **done** | ruff clean · 111 tests · real CVL: 216/94 writers, 70.0/30.0% samples, 0 overlap |
| T6 | Image normalisation (domain gap) | **done** | ruff clean · 200 tests · paper/ink/contrast spread 0 across conditions; ink% 38.0 -> 14.2 |
| T7 | Pack to a single LMDB file | **done** | ruff clean · batched writes · compaction 8.00 GB -> 24.9 MB |
| T8 | Dataset + collate | **done** | ruff clean · 19 tests · same-writer / different-word verified via style_keys |
| T9 | Checkpoint save/resume | **done** | ruff clean · 132 tests · resume is bit-identical to an uninterrupted run |
| T10 | Metrics: FID, CER, writer retrieval | **done** | FID 0.0000 self-check, floor 33.72 · CER 12.33% on real lines · retrieval **66.9% top-1** on 94 unseen writers |
| T11 | Experiment tracking + visual sample log | **done** | ruff clean · 157 tests · entity omri334jb configured |
| T12 | Colab end-to-end smoke run | **done** | T4, 2026-08-30: bit-identical resume, 6.1s Drive copy, 614 samples/s |

**Phase 2 — generation and its first real numbers.**

| ID | Task | Status | Verified by |
|----|------|--------|-------------|
| T13 | CVL line reader, with counted drops | **done** | ruff clean · 9,142 of 13,473 lines kept, total_seen matches the disk exactly |
| T14 | Line pack -> `cvl_lines_64.lmdb` | **done** | 9,142 lines, 309 writers, 127 MB compacted · `check_data.py` all green · rebuilt 2026-09-02 without the German passage |
| T15 | Re-measure FID / retrieval / CER on lines | **done** | CPU, 2026-09-09: FID floor 19.15 · writer 85.8% top-1, 94.4% top-5 · CER 11.45% over 300 lines · FID(real, same real) 0.0000 |
| T16 | Per-request token budget, then evaluate the generator | **done** | T4, 2026-09-09, two runs of 300. Best: FID 69.46 · writer 22.1% top-1 · CER gap +20.4% · 8.7% truncated |
| T17 | Confidence intervals, full-sample CER, saved analysis | **done** | 364 tests · every figure now carries a 95% spread · `analysis.npz` reconstructs a run's metrics exactly |
| T18 | Eruku adapter, and Eruku measured | **done** | T4, 2026-09-10, 128 min over 300: FID 80.74 [73.2, 88.3] · writer 5.3% [3.0, 8.0] · CER gap +13.0% · **zero empty outputs** |
| T19 | Sweep Eruku's guidance scale | **done, negative** | cfg 1.0 / 1.25 / 2.0 -> 10.0% / 5.3% / 6.7%, all intervals overlapping. cfg 2.0 clearly worse (15% truncated). Even cfg 1.0's upper bound is below Emuru's 22.1% |
| T20 | Several style lines as one reference | **done, thresholded** | 1 and 2 lines generate normally (0.89x, 0.85x of real width); 4 breaks Emuru (0.25x). The prefix outgrows what its stopping heuristic tolerates |
| T21 | Calibrate the style metric | **done** | A real line blurred 0.8px scores 12.2% against 96.8% untouched. The metric measures sharpness |
| T22 | HWD as the style metric | **done, floor superseded by T23** | 0.641 real / 0.721 blurred / 2.931 typeface, measured here at an unrecorded lines-per-writer |
| T23 | HWD's floor and ceiling measured inside every run | **done** | T4: Emuru HWD **2.00 [1.88, 2.13]** against real 0.86 and typeface 2.99 -- 53% of the way to no hand · fake check generated = typeface = 3.02 on GPU · real vs real 1.76 -> 0.54 from 1 to 10 lines per writer · per-writer HWD matches `HWDScore` to 2e-8 · 384 passed |
| T24 | HWD identity anchor: own writer against every other | **done** | T4: Emuru identity **56.5% [50.9, 61.9]** of real, own writer nearest 47.3% against chance 1.1% · fake on GPU -0.5% [-5.5, 4.4] · typeface gap -0.01 · run cells save to Drive themselves; cell 7b adds two style lines · 388 passed |
| T25 | Split a photographed page into lines | **done, one photo open** | 13 lines on 4 of 5 photos of Amri's page (angle, normal, shadow, WhatsApp) · `dim` fails upstream, `find_page` returns the whole frame -- strict xfail · 11 tests + 1 xfail |
| T26 | Dictated passage, two pages | **done, awaiting Amri's handwriting** | each page holds all 79 charset characters, every lowercase letter at least 3 times, lines of 35-44 characters · 7 tests |
| T27 | Emuru with two style lines, measured | **done, negative** | T4: identity 42.3% [36.2, 48.7] against 56.5% [50.9, 61.9] for one line · FID 90.64 against 67.70 · CER 59.6% against 30.4% · all three separate |

## Waiting on Amri

- [ ] IAM download into `data/raw/iam/`. **The FKI site is currently down** -- links do not
      load from either side. Not blocking: `tests/test_iam_real.py` activates by itself the
      moment `data/raw/iam/xml/*.xml` exists, and validates the reconstructed schema then.
      Priority order once reachable: `xml.tgz` (small, answers the schema question),
      then `ascii.tgz`, then `words.tgz` (~1.2GB).
- [x] 5 phone photos received. **They are on squared/graph paper** -- grid lines will be
      read as ink by naive binarisation. Ruled-line removal is now part of T6's scope.
- [x] `cvl-database-1-1.zip` downloaded and extracted. 99,904 cropped word images with
      transcriptions in their filenames; 98,179 usable after filtering.
- [ ] Weights & Biases username (the identifier only — never the API key)
- [x] Commit permission granted (messages record what was done and what changed)

## Open questions

- **Python version.** Local is 3.13.2 with torch 2.13.0+cpu (the CPU-only wheel, 122 MB).
  Colab on 2026-09-11: Python 3.13.15, T4, torch 2.11.0+cu128, and transformers 4.57.6
  after the install -- which therefore completed, `editdistance` included. Whether `hwd`
  imports there is what cell 2 now checks.
- **Architecture decided (2026-08-29): option C.** Start from a released zero-shot
  checkpoint (Emuru line) rather than training a generator from scratch. Amri approved.
  Consequences: no per-writer training; the generator produces variable-length lines, so
  word-to-line assembly leaves the layout engine's scope; IAM becomes evaluation data, not
  training data; generation resolution is 64px, upscaled afterwards. VATr++ stays as a
  documented fallback and comparison baseline. Surveys and their unverified-claims lists are
  in `docs/`.
- **IAM licence is non-commercial research use.** Fine for a portfolio project; a blocker if
  this ever ships as a product. Flagged early on purpose.

## Log

- **2026-09-14 — T24 and T27 measured: one line carries 56.5% of the identity, two
  carry less.** Cell 6 gave Emuru's first identity figure, 56.5% [50.9%, 61.9%] of what
  real lines carry, with the typeface at zero -- past half, where retrieval had put it
  at a quarter. Cell 7b joined two style lines into one image and fell below it on
  every measure that separated: identity 42.3% [36.2%, 48.7%], FID 90.64 against 67.70,
  CER 59.6% against 30.4%. The pass mark was set before the run. The real anchors did
  not move between runs, so the model did. A CER that doubles means the target text is
  not being written, which points at the joining rather than at the value of more
  evidence; the saved samples will say which.

- **2026-09-14 — T25 and T26: from a photographed page to lines, and a page made to be
  copied.** Amri's existing sample is one page photographed five ways -- 13 lines on
  squared paper, with a transcription split by sentence rather than by written line.
  Enough to build a splitter on; not enough to enrol from or test against. The
  splitter assigns connected components whole to the line their centre sits in, so a
  descender travels with its letter. Reaching 13 lines on four of the five photographs
  took four measured fixes: a strict ink threshold (ink sits at 0 after normalisation,
  surviving grid at 34-150); cutting straight runs out of the ink, so letters touching
  grid lines are no longer dropped as ruling; a border band against desk strips and
  dark page edges; and three components minimum per line, against punch holes.
  Hysteresis, to recover a lightly written "The", was built, measured and left off --
  it brings the grid back. The fifth photograph fails upstream in `find_page`, and its
  test is a strict xfail that says so. The passage is two pages of ordinary prose,
  each holding all 79 characters, every lowercase letter at least three times, every
  line within 44 characters; page 1 enrols, page 2 is held back so the system's version
  of it can be set beside Amri's.

- **2026-09-14 — T24: HWD can tell "not this writer" from "not real handwriting".**
  Every set is also measured against every other writer's reference, at no extra
  network pass. The gap -- how much farther the others are than the right writer -- is
  zero for output that is nobody's hand however clean or blurred, and the generated
  gap as a share of the real lines' is the identity figure, with an interval over
  writers. Locally the fake generator scored -0.2% [-6.0%, 5.8%], and real lines found
  their own writer nearest 97.7% of the time against a chance of 2.3%, so the anchor
  has range to measure in. Amri restated the goal the same day -- a system that works
  on his own pages, not a portfolio piece -- which makes the number of style lines the
  main axis: the product brings a page, the evaluation gave one line. The notebook
  gained cell 7b (two lines) and saves each run to Drive itself, after the VM holding
  the first HWD run was reclaimed unsaved. Also learned: NotebookEdit addresses cells
  by position, so an insert shifts every later edit onto the wrong cell; the notebook
  was rebuilt from HEAD by a script that checks each cell's heading before replacing.

- **2026-09-14 — the project's first trustworthy style figure.** Emuru on 300 held-out
  lines, T4: HWD 2.00 [1.88, 2.13] against 0.86 for the real target lines and 2.99 for
  the same texts in a typeface, all three against one shared reference in the same
  run -- 53% of the way from real handwriting to none. Against the old 0.641 constant
  it would have read 59%: the in-run floor matters most for a model close to real,
  which Emuru is not. FID and retrieval land where the earlier runs did. What the
  figure cannot yet say is how much of the distance is "not this writer" rather than
  "not real handwriting"; that needs a wrong-writer anchor, which the saved images
  allow without a GPU.

- **2026-09-11 — T23: HWD's floor and ceiling are measured inside every run.** Before
  spending the hour on cell 6, checked whether its HWD would be comparable to the 0.641
  floor. It would not: HWD compares per-writer means, and real lines against real lines
  read 1.76 at one line per writer and 0.54 at ten -- 1.06 under a 300-sample run's own
  sampling. 0.641 was measured at about seven or eight lines per writer by a script
  nobody committed, and would have made a perfect generator look nearly a fifth of the
  way to no hand at all. Every run now scores generated, real and typeface against one
  shared reference of real lines that are neither targets nor style lines, with
  intervals over writers; the per-writer computation matches `HWDScore` to 2e-8.
  Generated images are saved before any metric runs, and an HWD failure no longer
  takes CER with it. Also fixed cell 2, which imported `nib` inside the Colab kernel
  and found the clone directory as a namespace package instead.

- **2026-09-09 — T16 ran, and the project has its first measured result.** 300 lines
  in the hands of 94 unseen writers, T4, 53 minutes. FID 63.92 against a floor of
  19.15 and 254.29 for printed text; writer retrieval 20.5% top-1 against 85.8% for
  real and 1.7% for a generator with no style at all; CER gap +21.1 points. Read
  together: it produces handwriting (81% of the way from print to real) that is only
  partly the right person's (23% of the way from chance to real). `docs/phase2-first-evaluation.md`
  carries the full write-up.

  The run completed where the previous attempt died at 72 of 300. At request 41 the
  model declined to write, three re-draws failed, and it was excluded together with
  its ground truth instead of ending the run; four other requests were recovered by a
  retry. Every fix from the past fortnight was exercised by a real run rather than by
  a test.

  10.7% of the output was truncated, which is our bug and not the model's: the budget
  of 4.0 tokens per character covers about 92% of the data. Raised to 5.5, above the
  measured 99th percentile of 5.04, with the cap at 384 so it does not clamp ordinary
  text — at 5.5 a cap of 256 binds past 47 characters, and more than half the pack is
  longer. Real lines that do not fit their budget: 5.9% before, 0.7% after. The test
  for this is now derived from the pack rather than asserted on synthetic strings,
  because length and pixels-per-character are anticorrelated in real handwriting and
  no made-up string exposes that.

- **2026-09-09 — the German passage comes out, and the references with it.** A sample
  image from the pipeline check read `'Dann bist du deines Dienstes frey'`. The charset
  filter works on *characters*, and roughly three quarters of Goethe's Faust is pure
  ASCII, so 584 German lines had walked straight through it. Identified by the
  offending characters rather than by assumption, which mattered: text 3 loses lines to
  the same filter and is **English** — an article about an Austrian computer, the
  Mailuefterl, where only the lines naming it carry an umlaut. Excluding it too would
  have thrown away 2,051 sound English lines. Only passage 6 goes.

  The pack fell from 10,862 lines to 9,142, and **CER fell from 13.73% to 11.45%** —
  that 2.28 points is the size of the penalty we had been imposing on TrOCR for reading
  a language it was never trained on.

  Rebuilding exposed a hole in the references module itself: it identifies a pack by
  *filename*, and a rebuilt pack keeps its name, so numbers measured on 10,862 lines sat
  beside a pack of 9,142 looking entirely current. The file now records `pack_records`,
  and `evaluate_generator.py` refuses to start on a mismatch — before the 2.9 GB
  checkpoint download, not after an hour of GPU.

  Also fixed, found while checking the thinner pack: the retrieval gallery silently
  skipped writers with too few images **while keeping their queries**, which cannot
  match and quietly depressed the score. Those queries are now withheld and counted.

- **2026-08-31 — T15 done, and T16's code with it.** The line-level references are
  measured and committed: FID floor 19.06, writer retrieval 83.7% top-1 / 97.8% top-5,
  CER 13.73% over 300 lines (13.36% over 40 -- the cheap estimate held). It did not need a GPU — `check_metrics.py` generates nothing, and the
  220s-per-line figure belongs to generation alone. Two of the three moved far enough
  to change how a result reads: the FID floor nearly halved, so the word-level number
  would have made any generated set look almost twice as good as it is. They live in
  `references/`, which is a committed directory rather than part of the ignored
  `outputs/` — same reasoning as the committed writer split, since the numbers must
  survive the trip to whichever machine generates. T16's code landed too: a token
  budget from the text length, truncation counting, `--unit lines`, and a retrieval
  gallery that excludes the target samples so a match cannot be credited to shared
  content. `pyproject.toml` gained a `models` extra — the `transformers<5` pin was a
  decision recorded in prose and enforced nowhere, so a fresh Colab session would have
  installed 5.x and failed thirty minutes in. `notebooks/colab_eval.ipynb` runs the
  whole thing and holds no logic.

- **2026-08-31 — T14 done.** `cvl_lines_64.lmdb`: 10,862 lines, 309 writers, 148 MB
  compacted, built in 245s. One script packs both units — `build_index.py --unit
  lines|words` — because the two differ in exactly four things (reader, normaliser,
  key, header source) and those now sit in a `UNITS` table with everything else
  shared. A second near-identical script was the easier thing to write and the worse
  thing to read. `PackedWord` is now `PackedSample`; the record holds lines too and a
  name that lies is worse than a mechanical rename. The compacted upload copy is
  produced by the build rather than left as a step to remember — the module already
  said the shipped artefact is always compacted, and forgetting it once already cost
  an 8 GB upload. `check_data.py` reports both packs; only the word pack is required,
  since a machine set up for one job should not be told it is broken for lacking the
  other. Two tests exist specifically to catch a line pack built with the word
  scanner: mean width over 400px, and spaces in the text.

- **2026-08-31 — T13 done.** `nib.data.cvl_lines` reads CVL's 13,473 line crops and
  reassembles each transcription from the word filenames. 10,862 kept; the rest are
  counted by reason — 1,421 German, 1,132 with word-index gaps, 33 with no word files,
  25 from the writer CVL's own readme excludes. `total_seen` equals the file count on
  disk exactly. Word filenames are indexed in one unfiltered pass, keyed by
  (writer, text, line): filtering the index first would hide the very gaps it exists to
  find. Drop reasons are attributed in a fixed order — charset before incompleteness —
  so "incomplete" reports only English lines lost to data quality, and there is a test
  that pins that order. Added `normalise_line` beside `normalise_word`: the same
  mechanism, deliberately two names, because fixing the height destroys a word's
  relative scale and is exactly what makes lines comparable. Deleted `_cvl_lines` from
  `scripts/check_metrics.py`, which had reassembled transcriptions inline, filtered
  nothing, and drawn all 40 samples from one writer.

- **2026-08-28 — T0 done.** Package `nib-synth` installs editable as `nib`. Deviated from the
  brief's proposed layout: `src/nib/...` instead of `src/...`, so imports work from any working
  directory. torch deliberately excluded from base dependencies. Added `.gitattributes` for
  LF/CRLF, which would otherwise show every file as modified between Windows and Colab.
  Not committed — awaiting Amri.
