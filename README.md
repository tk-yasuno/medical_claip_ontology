# medical_claip (v0.2.0 Medical Ontology via CLAIP)

A port of the bridge-domain CLAIP architecture (an internal reference implementation,
v0.2.1: Image-Audio-Text contrastive
learning on a single bridge-repair-design meeting) to the medical domain, using only publicly, immediately
downloadable data. The goal is to let an experienced clinician's know-how be searched and cross-referenced
across clinical notes, voice, and images — as a foundation model that can support junior-doctor training,
symptom/procedure voice search, and image-based mental rehearsal before a procedure, especially relevant for
regions facing physician shortages.

Design details and the full decision log are kept in internal planning docs (v0.1.0 → v0.1.1 and v0.2.0),
not included in this public repository.

**This is a research / technical-demonstration MVP only. It must not be used for clinical judgement or
diagnostic support.**

## Why v0.2.0: same-domain audio and image

v0.1.0/v0.1.1 paired two *independent* public datasets from genuinely different clinical domains — PriMock57
(UK primary-care mock consultations: audio + transcript + clinical note) for an Audio-Text CLAIP branch, and
IU X-ray (US chest radiology: image + report) for an Image-Text CLAIP branch — connected only through a shared
frozen TextEncoder. Careful verification in v0.1.1 showed these two corpora share almost no vocabulary (only 4
terms — *heart, lung, hypertension, heart failure* — were ever mechanically confirmed to co-occur in both),
which capped cross-modal retrieval hit rates at roughly 1/4–1/7 regardless of how much training data was added.

For v0.2.0, we searched for a publicly, *immediately* downloadable dataset that pairs real human speech with
medical images in the same domain. The ideal candidate — PhysioNet's **"Eye Gaze Data for Chest X-rays"**
(built on MIMIC-CXR: real radiologist dictation audio + transcript + eye-gaze, aligned to the same images) —
exists, but requires PhysioNet's Credentialed Health Data License (CITI training, a signed Data Use Agreement,
and a multi-day review), which fails the "download right now" constraint of this MVP. No other public dataset
pairs real speech with medical images at the case level.

As a practical, honest substitute, v0.2.0 synthesizes narration audio for each IU X-ray report with a neural
TTS engine (Piper, isolated in its own `.venv-tts` to avoid dependency conflicts with the pinned torch stack in
`.venv-claip`). This guarantees image, audio, and text all describe the *exact same case* in the *exact same*
chest-radiology domain, restoring the bridge domain's original 3-modality design: a single `TripleProjectionHeads`
model trained with three pairwise InfoNCE losses (image–text, audio–text, image–audio) on one shared embedding
space, instead of two disjoint 2-modality branches. PriMock57 is dropped from the main v0.2.0 pipeline (its
code and v0.1.x results remain archived for reference) since its audio domain does not match IU X-ray's.

## Architecture (v0.2.0)

```
IU X-ray / Open-i HF mirror (MIT), 1,000-case subsample (seed=42)
        |
parse_iu_xray_reports.py  (1 report = 1 image)
        |
        +------------------------------+
        |                              |
synthesize_report_audio.py      encoders_medical.ImageTextEncoder
 (Piper TTS, .venv-tts,           (BiomedCLIP)
  16kHz mono narration of              |
  the SAME report_text)                |
        |                              |
encoders_medical.AudioEncoder.encode_audio_file()
 (Whisper encoder, same frozen encoder used in v0.1.x)
        |                              |
        +--------------+---------------+
                       |
          encoders_medical.TextEncoder (multilingual-e5-base, shared)
                       |
          build_triples_single_domain.py
       (image_emb, audio_emb, text_emb for the same 1,000 cases)
                       |
          train_triple_projection.py
   (TripleProjectionHeads: image_proj, audio_proj, text_proj;
    3 pairwise InfoNCE losses: image-text, audio-text, image-audio)
                       |
          eval_single_domain_retrieval.py
 (text->image, text->audio hit rate + global image<->audio
  closed-set self-retrieval Recall@1/@5 over all 1,000 cases)
```

## v0.2.0 results

| Metric | Value |
| --- | --- |
| Triples (single domain, same case) | 1,000 |
| Training loss (image-text / audio-text / image-audio), first → best | 6.95→1.18 / 6.92→0.55 / 6.96→0.88 |
| Text → Image hit rate (10 mechanically-selected queries) | 7/10 |
| Text → Audio hit rate (10 mechanically-selected queries) | 4/10 |
| **Global Image → Audio Recall@1** (closed-set, all 1,000 cases) | **0.9970** |
| Global Audio → Image Recall@1 | 0.9920 |
| Global Image → Audio Recall@5 | 0.9990 |

The global, query-independent Recall@1/@5 over all 1,000 cases is a much larger-sample and more reliable
indicator than the small, qualitative per-query hit-rate checks (same caveat as the bridge domain's
`eval_retrieval.py`: these are sanity checks on one dataset's closed set, not a statistically validated
Recall@K benchmark on held-out data). Because the audio is a *deterministic TTS rendering of the paired
report text* rather than independently recorded human speech, the audio–text alignment task is inherently
easier than it would be for real speech; the ~99.7% Recall@1 demonstrates that the CLAIP architecture and
training recipe work correctly for a genuine single-domain, same-case 3-modality dataset — not that the model
understands natural spoken language at that level. See
[output/medical_claip_v0.2.0_retrieval.json](output/medical_claip_v0.2.0_retrieval.json) for the raw results.

## v0.1.0 → v0.1.1 (archived two-branch approach)

Kept for reference: before the same-domain fix above, v0.1.x built two *independent* 2-modality CLAIP branches
and connected them only through a shared frozen TextEncoder.

<details>
<summary>v0.1.x architecture and results (click to expand)</summary>

```
PriMock57 (CC BY 4.0)                          IU X-ray / Open-i HF mirror (MIT)
doctor/patient-separated audio + manual        chest X-ray image + radiology report
transcript
        |                                                |
merge_speaker_segments (no diarization needed) parse_iu_xray_reports (1 report = 1 image)
        |                                                |
asr_medical (faster-whisper, English)          encoders_medical.ImageTextEncoder
        |                                       (BiomedCLIP, image only)
postprocess_medical (vocab_medical.yaml)                |
        |                                                |
wer_eval_medical (vs. text_gold)                         |
        |                                                |
        +--------------------+      +--------------------+
                       term_extraction_medical
         (PriMock57 notes highlights + IU X-ray report keywords
              -> Ollama classification -> domain_dict/medical_ontology.yaml)
                                     |
        +----------------------------------------------+
        |                                               |
build_triples_audio_text                      build_triples_image_text
  audio: encoders_medical.AudioEncoder           image: encoders_medical.ImageTextEncoder (BiomedCLIP)
  text : encoders_medical.TextEncoder            text : encoders_medical.TextEncoder (shared, multilingual-e5)
        |                                               |
train_projection_pair (audio,text)             train_projection_pair (image,text)
        |                                               |
        +-------------------+       +-------------------+
                             |       |
                   eval_cross_domain_retrieval
         (one query embedded once with the shared TextEncoder, then
          projected through each branch's own text_proj -> compare hit rates)
```

### Data acquisition

| Dataset | License | Method | Scale |
| --- | --- | --- | --- |
| PriMock57 | CC BY 4.0 | `git lfs clone` (babylonhealth/primock57) | 57 consultations (114 wav, 114 TextGrid, 57 notes) |
| IU X-ray / Open-i | CC BY-NC-ND 4.0 | Hugging Face mirror `MLforHealthcare/Indiana_University_Chest_X-ray_Collection` (the old NLM direct-download links were confirmed dead at implementation time; see Decisions) | v0.1.1: full 2,972 usable rows (v0.1.0 used a 1,000-row subsample) |

### ASR + domain-dictionary correction (WER evaluation, first 30 of 57 PriMock57 consultations, 3,730 segments)

Given the computational cost (~1–1.5 min/consultation with large-v3), scope was reduced to the first 30
consultations (all of day1 + day2, first 5 of day3) with user confirmation.

| Setting | Overall WER | Medical-term subset WER (410 segments) |
| --- | --- | --- |
| Baseline (no dictionary correction) | 0.3569 | 0.3286 |
| **Dict-only** (post-hoc correction of raw ASR output with `vocab_medical.yaml`, **adopted**) | **0.3568** | **0.3281** |
| `initial_prompt` re-decode + dict correction (**not adopted**) | 0.3846 | 0.3492 |

**Finding**: diffing 30 consultations' raw ASR output against PriMock57's manual transcripts surfaced 5 real
misrecognitions (e.g. `hayfever` → `high fever`, `microgynon` → `mark a garden`); applying them as a post-hoc
dictionary left WER essentially flat (no regression, slight improvement). Feeding the same terms to Whisper's
`initial_prompt` for re-decoding, however, increased silent/empty-hypothesis segments and *worsened* WER (a
short 5-term `initial_prompt` appears to have interfered with Whisper's silence detection). **Only post-hoc
dictionary correction is adopted**; the `initial_prompt` re-decode result is recorded as a negative finding —
being able to detect this regression quantitatively (thanks to PriMock57's ground-truth transcripts, which the
bridge domain never had) is itself a point in favor of using this dataset. The real-data misrecognition search
also surfaced 5 risky collisions with generic words or other valid medical terms (e.g. `asthma → acne`,
`right`/`arm`), excluded from auto-correction under the same policy as the bridge domain's `vocab.yaml`
(`src/vocab_merge_medical.py`'s `RISKY_GENERIC_VARIANTS`).

### Medical Ontology (`domain_dict/medical_ontology.yaml`)

Candidate terms from all 57 PriMock57 `notes/*.json` (`highlights` + `presenting_complaint`) and IU X-ray
report keywords, classified into 5 categories by a local LLM (qwen2.5:7b-instruct-q4_k_m; only
confidence="high" results merged).

| Category | Terms |
| --- | --- |
| Disease | 46 |
| Symptom | 96 |
| Procedure | 1 |
| Drug | 5 |
| Body-Part | 18 |
| **Total** | **166** |

### CLAIP contrastive training (2 independent branches)

| Branch | Dataset | Triples | Initial loss | Best loss |
| --- | --- | --- | --- | --- |
| Audio-Text | PriMock57 | v0.1.0: 3,730 (30 consultations) → **v0.1.1: 7,093 (all 57 consultations)** | 8.8790 | 5.5740 |
| Image-Text | IU X-ray | v0.1.0: 1,000 → **v0.1.1: 2,972 (full)** | 8.0425 | 3.5295 (800 epochs) |

The ASR/WER experiment stayed scoped to 30 consultations for cost reasons, but CLAIP's audio embedding
(`AudioEncoder.encode_audio_segment()`) is computed directly from the raw waveform and does not depend on ASR
output, so v0.1.1 extended the CLAIP training/retrieval corpus to all 57 consultations
(`build_triples_audio_text.py` gained a `segments.json` fallback).

### Cross-modal retrieval (qualitative sanity check)

**v0.1.1 fixed a root cause**: 5 of v0.1.0's 7 hand-picked query terms (pneumonia, cough, fever, chest pain,
emphysema, asthma, breathlessness) turned out to **never occur in any of the 2,972 IU X-ray reports** —
radiology reports describe imaging findings, not patient-reported symptoms — capping the achievable hit rate
at a structural 0% no matter how much image data was added. `find_shared_queries()` was added to
`eval_cross_domain_retrieval.py` to mechanically select only terms verified to co-occur in both corpora.

| Metric | v0.1.0 (7 hand-picked queries) | v0.1.1 (4 mechanically-verified queries: heart, lung, hypertension, heart failure) |
| --- | --- | --- |
| Image hit rate | 1/7 (14%) | **1/4 (25%)** |
| Audio hit rate | 3/7 (43%, partly from unanswerable queries) | 1/4 (25%) |

The fact that only 4 terms were ever confirmed to co-occur in both corpora is itself quantitative evidence
that PriMock57 and IU X-ray are genuinely different clinical domains — this is precisely what motivated v0.2.0
above. Scaling the Image-Text branch's data (1,000 → 2,972) barely moved the hit rate (the root cause was
query selection, not data volume); this negative result is recorded honestly rather than omitted.

</details>

## Reproduction

```powershell
# --- v0.2.0: single-domain 3-modality CLAIP (main pipeline) ---
# IU X-ray is downloaded automatically via huggingface_hub by parse_iu_xray_reports.py
.\.venv-claip\Scripts\python.exe src_claip\parse_iu_xray_reports.py --n-pairs 1000

# TTS narration synthesis (.venv-tts, isolated from .venv-claip's pinned torch stack)
python -m venv .venv-tts
.\.venv-tts\Scripts\python.exe -m pip install piper-tts
.\.venv-tts\Scripts\python.exe -m piper.download_voices en_US-lessac-medium --download-dir data\tts_voices
.\.venv-tts\Scripts\python.exe src_claip\synthesize_report_audio.py

# Build (image, audio, text) triples and train the 3-modality projection heads (.venv-claip)
.\.venv-claip\Scripts\python.exe src_claip\build_triples_single_domain.py
.\.venv-claip\Scripts\python.exe src_claip\train_triple_projection.py --triples-dir data\processed\image_text\claip_v0.2.0 --epochs 500

# Evaluate single-domain retrieval
.\.venv-claip\Scripts\python.exe src_claip\eval_single_domain_retrieval.py
```

<details>
<summary>v0.1.x reproduction (archived two-branch pipeline)</summary>

```powershell
git lfs install; git clone https://github.com/babylonhealth/primock57.git data/primock57

# Audio-Text branch (.venv-asr)
.\.venv-asr\Scripts\python.exe src\merge_speaker_segments.py --all
.\.venv-asr\Scripts\python.exe src\asr_medical.py --all --limit 30 --tag raw
.\.venv-asr\Scripts\python.exe src\term_extraction_medical.py
.\.venv-asr\Scripts\python.exe src\vocab_merge_medical.py --asr-tag raw
.\.venv-asr\Scripts\python.exe src\postprocess_medical.py --all --asr-tag raw --output-tag dictonly
.\.venv-asr\Scripts\python.exe src\wer_eval_medical.py --asr-tag raw --text-field text_asr --output output\wer_raw.json
.\.venv-asr\Scripts\python.exe src\wer_eval_medical.py --asr-tag dictonly --text-field text_corrected --output output\wer_dictonly.json

# Image-Text branch (.venv-claip)
.\.venv-claip\Scripts\python.exe src_claip\parse_iu_xray_reports.py --n-pairs 2972
.\.venv-claip\Scripts\python.exe src_claip\build_triples_image_text.py
.\.venv-claip\Scripts\python.exe src_claip\train_projection_pair.py --triples data\processed\image_text\claip\triples.json --emb-dir data\processed\image_text\claip --pair-names image,text --output-dir data\processed\image_text\claip --epochs 800

# Audio-Text CLAIP training (.venv-claip, all 57 consultations: no ASR needed, raw-waveform embedding only)
.\.venv-claip\Scripts\python.exe src_claip\build_triples_audio_text.py --all
.\.venv-claip\Scripts\python.exe src_claip\train_projection_pair.py --triples data\processed\audio_text\claip\triples.json --emb-dir data\processed\audio_text\claip --pair-names audio,text --output-dir data\processed\audio_text\claip --epochs 500

# Cross-modal retrieval evaluation (queries mechanically selected by find_shared_queries())
.\.venv-claip\Scripts\python.exe src_claip\eval_cross_domain_retrieval.py
```

</details>

## License

This project follows [LICENSE](LICENSE) (Apache License 2.0). Dataset
licenses apply separately: PriMock57 (CC BY 4.0), IU X-ray (CC BY-NC-ND 4.0) — non-commercial, research use
only.
