# Handwriting datasets: what we can actually use (2026-08-29)

We need human handwriting for **evaluation only**. Under the chosen architecture
(option C) the generator is a released checkpoint pretrained on synthetic fonts and
never trains on human data. Real handwriting is needed for the FID reference set,
the writer-retrieval metric, the CER baseline, and as genuine controls in the
deception study.

That reframes the licence question: a non-commercial research licence constrains
*published benchmark numbers*, not a shipped product, because no dataset and no
model derived from one would be distributed.

## Options

| Dataset | Licence | Writers | Writer ids | Availability | Notes |
|---|---|---|---|---|---|
| **CVL** | CC BY-NC 4.0 | 310 | **yes, by design** | **Zenodo, no registration** | Built *for* writer retrieval and identification. XML with per-word boxes. 4.2 GB full / 1.2 GB cropped |
| IAM | non-commercial + registration | 657 | yes (in XML / forms.txt) | **FKI site down** | The field standard, so published numbers are comparable |
| IAM mirror (Kaggle) | listed "Unknown"; IAM terms apply | 657 | **no** | downloadable now | 115k real word images + real `words.txt`. No XML, no `forms.txt` |
| **GNHK** | **CC-BY-4.0, commercial ok** | ? | **not verified — likely none** | goodnotes.com/gnhk | Camera-captured "in the wild" English. Modelled on scene-text datasets, which annotate regions rather than authors |
| IMGUR5K | non-commercial | ? | no | GitHub | Ruled out on licence |

## Decision

**CVL as the primary evaluation set.** It is downloadable today without a
registration wall, it carries writer ids as a first-class field because writer
retrieval is the task it was built for, and 310 writers is ample for evaluation
when we are not training. Cost: published FID numbers in the literature are
reported on IAM, so ours will not be directly comparable to papers. Accepted.

**GNHK as a permissive supplement** for the domain-gap work (T6): it is real
camera-captured handwriting, which is the actual target domain, and CC-BY means it
carries no restriction at all. Its lack of writer ids does not matter for that use.

**IAM if the site returns**, purely to report comparable numbers alongside ours.

## Unverified

- GNHK's image count, annotation fields, and whether any writer grouping exists.
  The GitHub README does not say and the paper PDF 404s. Confirm before relying on it.
- Whether the Kaggle mirror's `words_new.txt` differs from IAM's original beyond the
  dropped `number of components` column.

## A detail worth keeping

IAM's `words.txt` header documents the bad-segmentation flag as `er`, but the data
itself uses `err`. Our parser tests `!= "ok"` rather than `== "err"`, so both work.
