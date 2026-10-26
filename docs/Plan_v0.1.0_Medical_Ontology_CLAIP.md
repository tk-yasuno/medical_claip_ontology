Plan: v0.1.0 Medical Ontology via CLAIP Learning MVP — PriMock57×IU X-ray 二系統対照学習による医療ドメイン・言語汎用性実証

TL;DR
姉妹リポジトリ`0_LogBAK/pre-code_meet_mp4_claip`（v0.2.1）は、橋梁補修設計ドメインの単一Google Meet会議（44.9分、914発話セグメント）に対し、音声(Whisper encoder)・画像(rinna Japanese CLIP)・テキスト(multilingual-e5)の3モダリティをフリーズしたまま軽量射影ヘッドをInfoNCE対照学習させ（961トリプル、クロスモーダル検索ヒット率4/7）、かつontology裏付け（damage_cause_ontology由来6,745トリプル）によるASR用語辞書補正を実装済みである。
本計画では、CLAIPアーキテクチャ（encoders.py・ProjectionHeads・info_nce_pair・eval_retrieval.pyのパターン）を**コードはほぼそのまま移植し、データセットとエンコーダのみをドメイン・言語に合わせて差し替える**ことで、橋梁・日本語ドメインへの特化実装ではなく汎用フレームワークであることを実証する。具体的には、単一データセットで「音声×画像×テキスト」が揃った公開医療データが実質存在しないため、(a) PriMock57（英国プライマリケアの模擬診療57件、CC BY 4.0、音声+逐語録+診療ノート）でAudio-Text CLAIPペアを、(b) IU X-ray/Open-i（胸部X線7,470枚+読影レポート3,955件、CC BY-NC-ND 4.0、ログイン不要で直接ダウンロード可）でImage-Text CLAIPペアを独立に対照学習し、両者が同一の frozen TextEncoder（multilingual-e5-base）とyaml語彙フォーマットを共有することで、テキストクエリ経由の間接的クロスモーダル検索（音声側・画像側を同じクエリ語で検索）を実現する。
あわせて、PriMock57には橋梁版になかった「手動逐語録」という正解データがあるため、ASR生出力 vs ドメイン辞書補正後出力のWord Error Rate (WER) を定量比較し、医師の診療ノートの clinician-highlighted な医療用語（`highlights`フィールド）とX線レポートの所見語彙から Disease/Symptom/Procedure/Drug/Body-Part の5階層 Medical Ontology（`domain_dict/medical_ontology.yaml`）を構築する。最終的に、同一アーキテクチャが橋梁ドメイン(日本語)→医療ドメイン(英語)へそのまま転用できることを示し、CLAIPが特定ドメイン・言語に依存しない汎用基盤モデルであることのMVP実証とする。

Steps

Phase 1: 環境構築差分・ワークスペース初期化 (独立実行可能)
1. 既存`.venv-asr`・`.venv-claip`（本ワークスペースに橋梁版と同じ構成で導入済み: torch 2.11.0+cu128, faster-whisper 1.1.1, pyannote.audio 4.0.7, transformers 5.20.2, sentence-transformers 6.2.0 等）に差分パッケージを追加する
   - `.venv-asr`: `jiwer`（WER計算）を追加。`requirements-asr.txt`に追記
   - `.venv-claip`: `open_clip_torch`（BiomedCLIP読み込み用）を追加。`requirements-claip.txt`に追記。既存の`japanese_clip`・`easyocr`は医療パイプラインでは未使用のまま残す（削除不要、競合なし）
   - `python -c "from open_clip import create_model_from_pretrained; create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')"`でダウンロード・ロードを確認する（MITライセンス、ログイン不要）
2. ディレクトリ構成を作成する (parallel with 1)
   - `medical_claip/src/`（音声系, .venv-asr）, `medical_claip/src_claip/`（CLAIP対照学習系, .venv-claip）, `medical_claip/domain_dict/`, `medical_claip/data/`, `medical_claip/data/processed/{audio_text,image_text}/`
   - `0_LogBAK/pre-code_meet_mp4_claip/src/{asr.py,postprocess.py,vocab_merge.py}`と`src_claip/{encoders.py,train_claip_projection.py,eval_retrieval.py}`を移植元として参照し、Phase2以降でコピー＋改変する

Phase 2: データ取得 (独立実行可能, 2系統並列)
3. PriMock57を取得する (独立実行可能)
   - `git lfs install` → `git clone https://github.com/babylonhealth/primock57.git data/primock57`（音声はGit LFS管理、READMEに記載の手順通り）
   - 取得確認: `audio/`に`day{1-5}_consultation{NN}_{doctor,patient}.wav`、`transcripts/`に対応する`.TextGrid`（Praat long形式, IntervalTier, xmin/xmax/text, `<UNSURE>...</UNSURE>`・`<UNIN/>`タグ含む）、`notes/`に`day{D}_consultation{NN}.json`（`day`,`consultation`,`presenting_complaint`,`note`,`highlights`）が57件×doctor/patientの構成であることを確認する
4. IU X-ray/Open-iを取得する (独立実行可能, parallel with 3)
   - `wget https://openi.nlm.nih.gov/imgs/collections/NLMCXR_png.tgz`（約1.4GB）、`wget https://openi.nlm.nih.gov/imgs/collections/NLMCXR_reports.tgz`（約1.1MB）を`data/iu_xray/`へダウンロード・展開する
   - 展開直後、レポートXMLを1件実際に開いて`<AbstractText Label="FINDINGS">`/`<AbstractText Label="IMPRESSION">`/`<parentImage id="...">`の実スキーマを確認する（MeSH見出し要素の具体的なタグ名・有無も含め、Step 9実装前に必ず実データで確認する。ネット上の断片情報だけでは確定できないため）
   - ライセンス（CC BY-NC-ND 4.0: 非営利・改変物再配布禁止）をREADMEに明記し、本MVPでは埋め込み計算・研究目的利用のみに限定し、加工画像セットの再配布は行わない

Phase 3: Audio-Textブランチ — TextGrid解析・話者分離・ASR (.venv-asr, depends on 3)
5. 新規`src/textgrid_parse.py`を実装する
   - `parse_textgrid(path) -> list[{"start","end","text"}]`: Praat long形式をregexで解析（`xmin = ([\d.]+)`, `xmax = ([\d.]+)`, `text = "(.*)"`の3つ組を`intervals \[\d+\]:`ブロックごとに抽出。SudachiPy等の形態素解析は不要、英語は空白区切りのためトークナイザ依存部分を排除できる）
   - `<UNSURE>(.*?)</UNSURE>` → タグを剥がし中身のみ残す。`<UNIN/>`トークンは除去。結果が空文字列または空白のみの区間は破棄する（橋梁版`asr.py`の`min_segment_duration=0.2`相当のノイズ除去）
6. 新規`src/merge_speaker_segments.py`を実装する (depends on 5)
   - `day{D}_consultation{NN}_doctor.TextGrid`と`..._patient.TextGrid`を両方`parse_textgrid()`し、各区間に`speaker: "doctor"/"patient"`を付与して`start`でマージソートする
   - 出力`data/processed/audio_text/<consultation_id>/segments.json`: `[{"start","end","speaker","text_gold"}, ...]`（`text_gold`は手動逐語録、WER評価の正解ラベルとして温存する。pyannote diarizationは実行しない — PriMock57は話者チャンネル分離済み・人手逐語録付きのため不要、橋梁版Phase3aを代替・省略する）
7. 新規`src/asr_medical.py`を実装する (depends on 6, 橋梁版`src/asr.py`のtranscribe_segments()を移植・改変)
   - `language="en"`に変更（橋梁版は`language="ja"`固定）。各segmentの`speaker`に応じて対応する`{doctor,patient}.wav`から`[start,end]`を切り出してfaster-whisperでASRする
   - `MODEL_SIZE`は橋梁版と同じ`large-v3`をデフォルトとするが、`--model-size`で`medium`等へ軽量化できるようCLI引数化する（PriMock57は17h16m相当のチャンネル別音声があり、large-v3でのデコード時間が大きい場合の逃げ道を用意する。Decisions参照）
   - まず`day1`（最初の約10-12件）のみで動作確認し(Step 1での段階的スケーリング)、問題なければ`--consultations all`で57件全件を実行する
8. 新規`src/wer_eval_medical.py`を実装する (depends on 7)
   - `jiwer.wer(reference, hypothesis)`で、各segmentの`text_gold`と`asr_medical.py`の生出力（辞書補正前）のWERを計算し、`data/processed/audio_text/<consultation_id>/wer_raw.json`に保存する（corpus全体のmicro-averaged WERも算出）

Phase 4: Image-Textブランチ — IU X-rayレポート解析・BiomedCLIP埋め込み (.venv-claip, depends on 4, parallel with Phase 3)
9. 新規`src_claip/parse_iu_xray_reports.py`を実装する
   - `xml.etree.ElementTree`で`data/iu_xray/reports/*.xml`を走査し、各reportから`parentImage`の最初の1件（`id`属性, 複数ビューがある場合は前面像が先に列挙される慣行に従い先頭のみ採用、1 report = 1 imageに簡略化）、`FINDINGS`/`IMPRESSION`テキスト、Step4で確認したMeSH見出し相当の要素（存在すれば）を抽出する
   - `report_text = FINDINGS + " " + IMPRESSION`（両方空の場合はそのreportをスキップ）として`data/processed/image_text/pairs.json`: `[{"image_id","image_path","report_text","mesh_terms"}, ...]`を生成する
   - `random.seed(42)`で全件からユーザ確認済みの1,000件をサブサンプルする（`--n-pairs 1000`、デフォルト値）
10. 新規`src_claip/encoders_medical.py`を実装する (depends on Phase1 Step1, 橋梁版`src_claip/encoders.py`を移植)
    - `AudioEncoder`・`TextEncoder`クラスは**無変更で流用**する（whisper-small encoderもmultilingual-e5-baseも元々多言語/英語対応のため、橋梁版の日本語専用ロジックが入っていないコア部分はそのまま使える）
    - `ImageTextEncoder`のみ`rinna/japanese-clip-vit-b-16`から`open_clip.create_model_from_pretrained("hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")`に差し替える。`encode_image()`/`encode_text()`のインターフェースは橋梁版と同一シグネチャを維持し、`build_triples_image_text.py`から見た差し替えコストをゼロにする
11. Step9のpairs.jsonに対し`ImageTextEncoder.encode_image()`/`encode_text()`で埋め込みを計算し、`data/processed/image_text/image_emb.npy`・`text_emb.npy`・`pairs.json`（画像パス・report_text付き)を保存する

Phase 5: Medical Ontology構築（5階層: Disease/Symptom/Procedure/Drug/Body-Part） (depends on Phase3 notes, Phase4 reports)
12. 新規`src/term_extraction_medical.py`を実装する（橋梁版`term_extraction.py`/`ontology_vocab_import.py`のclassify-and-mergeパターンを移植、ただし日本語かな/漢字変換ロジックは完全に削除）
    - 音声側候補: `data/primock57/notes/*.json`全57件の`highlights`リスト全件＋`presenting_complaint`を集約し、頻度集計する（クリニシャン自身が重要医療情報として手動ハイライトした語句であり、会議transcriptから機械抽出する橋梁版term_extraction.pyより既に裏付けが強い。追加のNLP句抽出は不要）
    - 画像側候補: Step9の`mesh_terms`（存在する場合）または`IMPRESSION`文からの簡易キーワード頻度抽出（`pneumonia`, `cardiomegaly`, `effusion`, `atelectasis`, `consolidation`, `opacity`, `nodule`, `emphysema`等の胸部X線頻出所見語を正規表現候補リストとして用意し、出現有無をカウントする。LLMに所見文から新規診断を推論させることはしない — Decisions参照）
    - 候補語を局所Ollama LLM（`qwen2.5:7b-instruct-q4_k_m`、橋梁版と同一モデル）に投げ、「(1) Disease/Symptom/Procedure/Drug/Body-Partのいずれに分類されるか (2) 医療ドメイン語彙として適切か(はい/いいえ) (3) 信頼度(高/中/低)」を判定させるプロンプト（`TERM_CLASSIFY_PROMPT_TEMPLATE`の医療版、橋梁版の「橋梁工学専門用語か」を「5分類のいずれかに該当する医療専門語か」に置換）
    - 信頼度「高」の語のみ`domain_dict/medical_ontology.yaml`（`category: [term1, term2, ...]`の5キー構造）へマージする
13. 新規`src/vocab_merge_medical.py`を実装する (depends on 7の生ASR出力とtext_goldの差分)
    - Step7のASR生出力とPriMock57手動逐語録（`text_gold`）を`difflib.SequenceMatcher`等で単語単位diffし、医療専門語（Step12で分類済みの語彙）が関与する実際の置換誤り（例: "gastroenteritis"→"gastro and triters"のような whisper の実測誤認識）のみを抽出する
    - 抽出した実測バリエーションを`domain_dict/vocab_medical.yaml`（`正式表記: [誤認識バリエーション]`、橋梁版vocab.yamlと同じflat形式）へ人手レビュー後に追記する（橋梁版の「実データ検証で判明した方針」と同様、一般語との衝突リスクがある変種は除外する）

Phase 6: ASR辞書補正の再実行・WER改善評価 (.venv-asr, depends on 13)
14. `src/postprocess_medical.py`を実装する（橋梁版`postprocess.py`のapply_domain_dict()を移植・簡略化）
    - 英語はスペース区切りのため、SudachiPyトークナイザは不要。`re.compile(r'\b' + re.escape(variant) + r'\b', re.IGNORECASE)`による単語境界マッチのみで`domain_dict/vocab_medical.yaml`の変種を正式表記へ置換する（橋梁版の「2文字以下は危険」という日本語特有の部分文字列問題は英語の単語境界マッチでは発生しないため、2パス方式は不要と判断する）
15. `src/asr_medical.py`を`initial_prompt=build_initial_prompt(vocab_medical.yaml)`付きで再実行し、`postprocess_medical.py`で補正後、`wer_eval_medical.py`を再実行して`wer_corrected.json`を生成する
16. `wer_raw.json`と`wer_corrected.json`を比較し、corpus全体のWERが悪化していないこと（corrected WER <= raw WER、医療専門語を含むセグメントでの改善幅）を確認するレポート`output/primock57_wer_comparison.md`を生成する（橋梁版READMEのv0.1.0→v0.2.0比較表と同じ形式）

Phase 7: CLAIP対照学習 — Audio-Text / Image-Textの2ブランチ独立学習 (depends on Phase3,4,5)
17. 新規`src_claip/build_triples_audio_text.py`を実装する（橋梁版`build_triples.py`のsegment-centric方式を移植）
    - Step6の`segments.json`の各utterance音声区間を`AudioEncoder.encode_audio_segment()`で埋め込み、同じ`consultation_id`の診療ノート全文（`notes/*.json`の`note`フィールド）を`TextEncoder.encode_text()`で埋め込む
    - 1 consultationにつき audio embeddingは utterance数ぶん、text embeddingは1件のみ生成し、同一consultation内の全audioが同じtext embeddingと正ペアを成すトリプルとして`triples_audio_text.json`・`audio_emb.npy`・`text_emb.npy`を保存する（橋梁版の「1segmentに複数frameが対応」を反転させた「1noteに複数audioが対応」構造）
18. 新規`src_claip/build_triples_image_text.py`を実装する (parallel with 17)
    - Step11で計算済みの`image_emb.npy`・`text_emb.npy`・`pairs.json`をそのまま`triples_image_text.json`としてラップする（1 report = 1 imageのため1:1ペア、trivialなラッパー）
19. 新規`src_claip/train_projection_pair.py`を実装する（橋梁版`train_claip_projection.py`の`ProjectionHeads`/`info_nce_pair()`を2モダリティ版へ簡略化: `PairProjectionHeads(a_dim, b_dim, proj_dim=256)`が`a_proj`・`b_proj`の2つの線形層のみを持ち、`info_nce_pair()`は橋梁版からそのままコピーする）
    - `python src_claip/train_projection_pair.py --triples data/processed/audio_text/triples_audio_text.json --pair-names audio,text --output-dir data/processed/audio_text/claip --epochs 300 --seed 42`
    - `python src_claip/train_projection_pair.py --triples data/processed/image_text/triples_image_text.json --pair-names image,text --output-dir data/processed/image_text/claip --epochs 300 --seed 42`
    - 両ブランチとも`proj_dim=256`・`tau=0.07`で統一し（橋梁版のデフォルトを踏襲）、学習曲線（best/final loss）を`output/`配下にログ出力する

Phase 8: クロスモーダル検索評価・CLAIP汎用性実証 (depends on 17,18,19)
20. 新規`src_claip/eval_cross_domain_retrieval.py`を実装する（橋梁版`eval_retrieval.py`のcosine_topk()パターンを移植、2ブランチ分を1スクリプトで実行）
    - クエリ語は`domain_dict/medical_ontology.yaml`のDisease/Symptomカテゴリから、音声側(PriMock57)・画像側(IU X-ray)の双方の語彙頻度表に重なりがある語（例: pneumonia, cough, chest pain, fever等、Step12のカテゴリ分類結果から機械的に抽出）を既定クエリとする
    - 同一の`TextEncoder.encode_text(query)`生埋め込みを、(a) audio_textブランチの`text_proj`で射影してaudio_emb最近傍を検索、(b) image_textブランチの`text_proj`で射影してimage_emb最近傍を検索し、ヒット有無（top1のテキスト側に query語が含まれるか）を両ブランチ別に記録する
    - 出力`output/medical_claip_cross_domain_retrieval.json`: `{"query", "audio_hit", "audio_top_text", "image_hit", "image_top_report"}`のリスト。橋梁版eval_retrieval.pyと同じ文言で「単一データセットの閉集合に対する定性的サニティチェックであり、統計的に検証されたRecall@Kベンチマークではない」旨を明記する
21. `README.md`（medical_claipワークスペース直下、新規作成）に本MVPのアーキテクチャ図・v0.1.0結果サマリ（WER改善表、medical_ontology.yamlの5カテゴリ別語数、audio-text/image-textそれぞれのCLAIP学習曲線、クロスモーダル検索ヒット率）を橋梁版READMEと同じ構成で記載する

Relevant Files
- `0_LogBAK/pre-code_meet_mp4_claip/src_claip/encoders.py` — 移植元。AudioEncoder/TextEncoderは無変更流用、ImageTextEncoderのみBiomedCLIPへ差し替える参照実装
- `0_LogBAK/pre-code_meet_mp4_claip/src_claip/train_claip_projection.py` — `ProjectionHeads`/`info_nce_pair()`の2モダリティ簡略版(`train_projection_pair.py`)のロジック移植元
- `0_LogBAK/pre-code_meet_mp4_claip/src_claip/eval_retrieval.py` — `eval_cross_domain_retrieval.py`のcosine_topk()・出力フォーマット・免責文言の移植元
- `0_LogBAK/pre-code_meet_mp4_claip/src/asr.py`, `postprocess.py`, `vocab_merge.py` — 医療版(`asr_medical.py`, `postprocess_medical.py`, `vocab_merge_medical.py`)の移植元。日本語形態素解析(SudachiPy)・かな読みロジックは医療版では全て不要になる点を明示的に差分化する
- `0_LogBAK/pre-code_meet_mp4_claip/domain_dict/vocab.yaml` — `domain_dict/vocab_medical.yaml`のフォーマット踏襲元（flat `正式表記: [variants]`）
- `data/primock57/` — PriMock57本体（git lfs clone、CC BY 4.0）: `audio/*.wav`, `transcripts/*.TextGrid`, `notes/*.json`
- `data/iu_xray/` — IU X-ray/Open-i本体（直接wget、CC BY-NC-ND 4.0）: `NLMCXR_png/*.png`, `reports/*.xml`
- `domain_dict/vocab_medical.yaml` — 新規、ASR誤認識補正辞書（英語, 実測バリエーションのみ）
- `domain_dict/medical_ontology.yaml` — 新規、本計画の中心成果物。5階層(Disease/Symptom/Procedure/Drug/Body-Part) Medical Ontology
- `src/textgrid_parse.py`, `merge_speaker_segments.py`, `asr_medical.py`, `wer_eval_medical.py`, `postprocess_medical.py`, `vocab_merge_medical.py`, `term_extraction_medical.py` — 新規、音声系(.venv-asr)
- `src_claip/encoders_medical.py`, `parse_iu_xray_reports.py`, `build_triples_audio_text.py`, `build_triples_image_text.py`, `train_projection_pair.py`, `eval_cross_domain_retrieval.py` — 新規、CLAIP系(.venv-claip)

Verification
- Step3/4: `git clone`後に`data/primock57/notes/*.json`が57件、`data/iu_xray/reports/*.xml`と`NLMCXR_png/*.png`がそれぞれ3,955件/7,470件程度存在することを`ls`/`Get-ChildItem`件数で確認する
- Step7/15: `python src/asr_medical.py --consultations day1 --model-size large-v3` → 空出力セグメント数が全体の10%未満であることを確認後、`--consultations all`で57件を実行する
- Step16: `python src/wer_eval_medical.py --compare wer_raw.json wer_corrected.json` → corpus全体のcorrected WER <= raw WERであることを数値で確認する（医療専門語を含むセグメントに限定したサブセットWERも併記し、改善幅が正であることを確認する）
- Step12: `domain_dict/medical_ontology.yaml`の5カテゴリ(Disease/Symptom/Procedure/Drug/Body-Part)全てに最低1件以上の語が登録されていることを確認する（空カテゴリがあればLLM分類プロンプトまたは候補語抽出ロジックを見直す）
- Step19: `train_projection_pair.py`実行ログで、audio_text・image_text両ブランチともbest epoch lossがinitial epoch lossより低下していることを確認する
- Step20: `python src_claip/eval_cross_domain_retrieval.py` → `output/medical_claip_cross_domain_retrieval.json`が生成され、クエリ数に対しaudio_hit/image_hitのいずれかが1件以上Trueであることを確認する（橋梁版と同じ「定性的サニティチェック」水準）
- データ整合性: `triples_audio_text.json`の件数が`segments.json`の総utterance数と一致、`triples_image_text.json`の件数がStep9で指定した1,000件と一致することを確認する

Decisions
- 単一の公開データセットに「音声×画像×テキスト」が揃ったものが実質存在しないため、3モダリティ同時トリプルは採用せず、PriMock57(Audio-Text)とIU X-ray(Image-Text)の2つの独立した2モダリティCLAIPブランチを構築し、同一のfrozen TextEncoder(multilingual-e5-base)とyaml語彙フォーマットを共有することでクロスモーダル検索の接続点とする。橋梁版の「1会議内でのimage-audio-text同時トリプル」よりも現実的な妥協だが、公開データのみでCLAIPの汎用性を示すというMVPの目的には合致する
- 画像×テキスト側データセットはIU X-ray/Open-iを採用（MultiCaRe/MedPix 2.0は見送り）: ログイン不要の直接wgetでダウンロード可能、ライセンスが明確(CC BY-NC-ND 4.0)、既存研究での利用実績が豊富で解析の見通しが立てやすいため。MultiCaReはライセンスがファイル単位で混在し、MVPの検証コストが高いため今回は見送り、v0.2以降の拡張候補とする
- PriMock57側はpyannote diarizationを実行しない: 音声がdoctor/patientの別チャンネルで提供され、かつ手動逐語録(TextGrid)で発話区間が既に確定しているため、橋梁版Phase3a相当のステップは不要と判断し計算コストを削減する
- 画像側の候補語抽出にLLMへ所見文からの新規診断推論をさせない: IMPRESSION/FINDINGS文からLLMに「この患者の診断は何か」を推論させることは臨床的判断の代行になり医療安全上不適切なため、既存のMeSH見出し相当の構造化情報または事前定義した胸部X線頻出所見語の出現検知のみを候補とし、LLMの役割は橋梁版と同じ「語彙分類（5カテゴリのどれか）」に限定する
- `domain_dict/vocab.yaml`のSudachiPy形態素解析・かな読み/漢字変換ロジックは医療版(`vocab_medical.yaml`)に移植しない: 英語はスペース区切りのため`\b...\b`の単語境界正規表現のみで安全に置換判定でき、日本語固有の「部分文字列誤マッチ」問題（受けた→受桁 等）が構造的に発生しないため
- IU X-rayは1 report = 1 image（最初のparentImageのみ採用）に簡略化する: 複数ビュー画像と1レポートの対応関係をXMLのみから厳密に判定するロジックは本MVPのスコープ外とし、先行研究での簡略化慣行に倣う
- v0.1は英語データ(PriMock57: 英国英語, IU X-ray: 米国英語)で技術実証を行い、日本語臨床データへの適用は見送る: ただしTextEncoder(multilingual-e5-base)は最初から多言語対応であり、橋梁版(日本語)→本計画(英語)への移植がコード変更ほぼゼロで成立したこと自体が「ドメインだけでなく言語にも依存しない」というCLAIPの汎用性実証の一部であると位置づける
- ユーザー確認に基づき、LLMによる新規医療語彙の自動マージ(term_extraction_medical.py)を実施する。ただし分類対象はPriMock57の`highlights`（医師自身が手動ハイライトした語句）とX線所見の構造化/既定キーワードのみに限定し、LLMが自由テキストから独自に医療用語を発案することはない（橋梁版term_extraction.pyの「用語がドメイン語彙かどうかを判定する」役割を厳密に踏襲し、「何が正しい診断か」は一切判定させない）
- IU X-rayはCC BY-NC-ND 4.0（非営利・改変物再配布禁止）のため、本MVPでは埋め込み計算・検索実験等の研究利用にとどめ、加工済み画像セットやサブサンプルの再配布は行わない
- 【実装時のスコープ調整】PriMock57は全57件のうち、ASR・WER評価・CLAIP Audio-Textブランチの学習対象を先頭30件（day1全15件+day2全10件+day3先頭5件、`segments.json`のソート順で決定）に縮小した。large-v3でのASRデコードが1件あたり約1〜1.5分かかり、57件全件では計算時間がMVPの検証サイクルに対して過大と判断したため（ユーザー確認済み）。Medical Ontology構築（Phase5, `term_extraction_medical.py`)は`notes/*.json`全57件のhighlightsを引き続き活用する（テキストのみでASRデコードを伴わず低コストなため）
- IU X-ray/Open-iの旧NLM直リンク（`openi.nlm.nih.gov/imgs/collections/*.tgz`）は実装時点で既に廃止されており、"There seems to be an error"というHTMLエラーページが返ることを確認した（NLM側がopeni-vip.nlm.nih.govのAngular SPAへ移行済み）。同一データセットのHugging Faceミラー`MLforHealthcare/Indiana_University_Chest_X-ray_Collection`（MITライセンス、ログイン不要、parquet形式でimage/question/report列）へ切り替えた。データ内容は同一（IU X-ray 7,470枚/3,955レポート由来）で、XML+PNG直接パースからparquet読み込みへ実装が簡略化された

Further Considerations
- クエリ語の音声側(primary care主訴)と画像側(胸部X線所見)の意味的重なりが薄い懸念（内科外来の一般的訴えと放射線科の専門所見は臨床文脈が異なる）
  推奨: v0.1ではpneumonia/cough/chest pain等、両データセットで実際に出現する語に限定してクロスモーダル検索クエリを機械選定し、重なりが乏しい場合はその事実自体を結果として正直に記載する（無理に高ヒット率を狙わない）
- 真の3モダリティ（video+audio+image+text）同時トリプルによるCLAIP学習（ユーザーの「手術準備のイメージトレーニング」構想に対応）
  推奨: v0.2以降、公開の教育用診療動画（YouTube医学教育チャンネル等、ライセンス確認済みのもの）からframe_extract/frame_ocr/frame_caption（橋梁版src_claipをほぼそのまま流用可能）で自前データセットを構築し、橋梁版と同じ3モダリティ構成に拡張する
- テキスト・画像の生成によるイメージトレーニング支援（ユーザー構想）
  推奨: CLAIPは対照学習（検索・埋め込み整合）であり生成モデルではないため、v0.1はretrieval実証にとどめる。生成機能は共有Embedding空間を条件とした拡散モデル等への接続としてv0.3以降の別計画で扱う
- UMLS/SNOMED CT等の正式な医療オントロジーへのリンク付け（本MVPのmedical_ontology.yamlは5カテゴリの簡易自前分類）
  推奨: v0.1は自前の5階層分類で十分とし、UMLSライセンス取得後にentity linking精度を上げる拡張をv0.2の課題とする
- `asr_medical.py`のlarge-v3モデルでPriMock57全57件(17h16m相当、doctor/patient channel合算)をデコードする実行時間が長大になるリスク
  推奨: Step7のとおりday1サブセットでまず動作確認し、実測デコード時間からGPU時間を見積もった上で全件実行の可否を判断する（必要なら`--model-size medium`へ切替可能なようCLI引数化済み）
- train_projection_pair.pyの全バッチ学習（ミニバッチなし）がPriMock57の推定3,000〜6,000トリプル規模でメモリ・時間的に許容範囲か
  推奨: logits行列は(N×N)でN=6,000でも float32 で約144MBとGPU 16GBに収まる計算のため、v0.1はミニバッチ化を見送り、実測に応じてv0.2でDataLoader化を検討する
- 本MVPは研究・技術実証目的であり、実際の臨床判断・診断支援に使用するには医師による内容検証・妥当性確認が別途必須である
  推奨: README・medical_ontology.yamlの先頭コメントに「研究用MVP、臨床使用不可」の注記を明記する

---

## v0.1.1 追記: Image-Textブランチのフルデータ化とクロスモーダル検索の根本原因修正

v0.1.0実装結果（Image-Text branch 1,000ペアサブサンプル時点）では、クロスモーダル検索のimage hit rateが1/7（14%）と低かった。
ユーザーからの「IU X-rayを2,972件フルに使い、image hit rateを改善したい」という要求を受けて調査したところ、根本原因は
**学習データ量ではなくクエリ語選定方法にあった**ことが判明した: v0.1.0の`DEFAULT_QUERIES`（pneumonia, cough, fever,
chest pain, emphysema, asthma, breathlessness）のうち5/7語（cough, fever, chest pain, asthma, breathlessness）は、
2,972件のIU X-rayレポート全文を検索しても**1件も出現しない**ことが判明した（放射線科レポートは画像所見を記述するもので、
患者申告症状を記述しないため）。つまりこれら5語はヒット率の天井が構造的に0%であり、画像データを何件に増やしても
改善しようがなかった。

### 実施した修正

1. **Image-Textブランチのデータを1,000件→2,972件（HFミラーから取得可能な全件）へ拡大**（`parse_iu_xray_reports.py --n-pairs 2972`）。
   再学習（epochs=800）でbest loss 8.04→3.53まで収束を確認。
2. **`eval_cross_domain_retrieval.py`に`find_shared_queries()`を新規実装**し、クエリ語を手動推測ではなく、
   Audio-Textブランチのnote_text・Image-Textブランチのreport_textの両方に実際に出現する語をメカニカルに検証して選定するよう変更した
   （`--queries`省略時のデフォルト動作。`domain_dict/medical_ontology.yaml`のDisease/Symptom/Body-Part/Procedure/Drug全カテゴリを候補に、
   出現回数でソートした上位N語を採用する）。
3. **Audio-Textブランチの音声・テキストコーパスを30件→全57件へ拡大**（`build_triples_audio_text.py`に`segments.json`フォールバックを追加）。
   CLAIPの音声埋め込みは`AudioEncoder.encode_audio_segment()`で生波形から直接計算されASR文字起こし結果に依存しないため、
   ASR/WER実験の対象を30件に限定した後も、CLAIP対照学習・検索コーパスとしては全57件分の音声+診療ノートを利用できる
   （ASR/WER実験のスコープ30件は維持し、CLAIP学習・検索のスコープのみ全57件へ切り離した）。この結果、音声側トリプル数は
   3,730→7,093に増加し、`find_shared_queries()`で検出される真の共有語も3語（heart, Hypertension, Heart failure）から
   4語（+ lung）に増えた。

### v0.1.1 結果

| 指標 | v0.1.0 | v0.1.1 |
| --- | --- | --- |
| Image-Textトリプル数 | 1,000 | **2,972** |
| Image-Text best loss | 1.2319 (300 epochs) | **3.5295** (800 epochs, より難しい負例集合のため絶対値は上昇するが収束は確認) |
| Audio-Textトリプル数 | 3,730 (30 consultations) | **7,093** (57 consultations) |
| クロスモーダル検索クエリ選定方法 | 手動推測（7語中5語が検証対象コーパスに0件） | **メカニカル選定**（両コーパスに実在する語のみ） |
| 選定クエリ数 | 7（うち5語はヒット不可能） | 4（全語が両コーパスに実在） |
| Image hit rate | 1/7 (14%) | **1/4 (25%)** |
| Audio hit rate | 3/7 (43%, うち一部は無関係クエリ) | 1/4 (25%) |

Image hit rateは測定方法自体をクエリの答えられなさ（0%天井）という欠陥を取り除いた上で1/4（25%）に改善した。
Audio hit rateは一見低下したように見えるが、v0.1.0の3/7はcough/chest pain/breathlessness等、画像側では原理的に
ヒットし得ない（かつ監査により偶然音声側ではヒットしていた）クエリを含んでおり、v0.1.1の4語はいずれも両コーパスに
実在することが検証済みの、より公正な評価対象である。4クエリという小規模な評価セットであるため1/4という数値自体の
分散は大きく、定性的サニティチェックの域を出ない点は v0.1.0 と変わらない。

### Decisions（v0.1.1追加分）
- Image-Textブランチのデータ拡大（1,000→2,972件）そのものはhit rateをほぼ変化させなかった（根本原因はクエリ選定であったため）。
  このネガティブな実験結果も「データを増やせば必ず改善する」という安易な仮説を否定する知見として正直に記録する
- `DEFAULT_QUERIES`という名前の手動推測クエリ定数は廃止し、`find_shared_queries()`によるメカニカル選定をデフォルト動作とした。
  `--queries`引数での手動上書きは引き続き可能（デバッグ・特定語の確認用途）
- Audio-Textブランチの「CLAIP学習・検索コーパス」と「ASR/WER実験」のスコープを分離した: 前者は全57件（ASR文字起こし不要、
  生波形埋め込みのみ）、後者は30件（計算コストの都合、ユーザー確認済み）のまま維持する

### Further Considerations（v0.1.1追加分）
- 真に共有される語彙が4語（heart, lung, Hypertension, Heart failure）しかないという事実は、PriMock57（英国プライマリケア）と
  IU X-ray（米国胸部X線）が臨床的に別ドメインであることの定量的証拠である
  推奨: v0.2でより語彙重複の大きいデータセットの組み合わせ（例: 両方とも胸部疾患を扱うデータセット同士）を検討するか、
  "symptom -> finding" のような同義語マッピング（例: "breathlessness" <-> "dyspnea"）を医療オントロジー側に追加し、
  表層形の一致ではなく概念レベルでの一致を評価できるようにする

