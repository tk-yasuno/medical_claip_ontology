Plan: v0.2.0 Medical Ontology via CLAIP Learning MVP — 単一ドメイン(胸部放射線科)3モダリティCLAIPへの再設計

TL;DR
v0.1.1までの実装（docs/Plan_v0.1.0_Medical_Ontology_CLAIP.md）は、単一データセットで「音声×画像×テキスト」が揃った公開医療データが
実質存在しないという制約から、PriMock57(Audio-Text, 英国プライマリケア)とIU X-ray(Image-Text, 米国胸部X線)という2つの独立した
2モダリティCLAIPブランチを構築した。v0.1.1での定量検証の結果、この2ブランチ間で実際に両コーパスに出現することを確認できた共有語彙は
わずか4語（heart, lung, hypertension, heart failure）にとどまり、クロスモーダル検索ヒット率は1/4(image)・1/4(audio)という低い値に
留まった。原因はデータ量ではなく、2つのデータセットが真に異なる臨床ドメイン（プライマリケアの症状表現 vs. 放射線科の所見記述）で
あったことだと特定済みである。
本計画では、ユーザーの「audioとimageがsame domainとなるようにデータを探す」という要求に対し、(1) PhysioNetの
"Eye Gaze Data for Chest X-rays"（MIMIC-CXR由来、放射線医の実音声口述+画像+書き起こしが完全に紐づく理想的データセット）を発見したが
Credentialed Health Data License（CITI研修+DUA締結+審査、承認まで数日）が必須で「即座にダウンロード可能」というMVPの前提に反するため
採用を見送り、(2) 代わりにIU X-rayのレポートテキストをニューラルTTS（Piper, 独立した.venv-tts環境）でナレーション音声化し、
image(X線画像)×audio(TTS音声)×text(レポート)を**同一症例・同一ドメイン**の3モダリティトリプルとして再構築した。これにより、
橋梁版CLAIPの本来のアーキテクチャ（ProjectionHeadsの3ペア同時InfoNCE対照学習: image-text, audio-text, image-audio）を完全な形で
医療ドメインに移植でき、PriMock57ブランチ（非同一ドメイン）はユーザー確認のもとv0.2.0のメインパイプラインから除外した。
結果、グローバルなimage↔audio自己検索Recall@1が99%超、text→image/text→audioの定性的クエリヒット率もv0.1.1の1/4から
7/10・4/10へ大幅に改善し、同一ドメイン化が決定的に重要なレバーであったことを実証した。

Steps

Phase A: 同一ドメイン音声データの探索 (独立実行可能)
1. 「audio×imageが同一ドメイン」の公開データセットを調査する
   - PhysioNet "Eye Gaze Data for Chest X-rays" (egd-cxr, MIMIC-CXR由来1,083症例、放射線医口述音声+書き起こし+視線データ完備)を発見。
     ただしCredentialed Health Data License必須（CITI研修証・DUA・審査、数日〜）のため「即座にダウンロード可能」条件を満たさず不採用
   - ICBHI Respiratory Sound Database（肺音聴診録音）は画像とのペアが存在せず不採用
   - 結論: 症例レベルで音声×画像が紐付く即座にダウンロード可能な公開データは実質存在しないため、TTS音声合成による代替を採用する
     （ユーザー確認済み）
2. TTS音声合成エンジンを選定・導入する (parallel with 1)
   - `piper-tts`（ONNX実行、オフライン、ニューラル音声、torch非依存）を新規`.venv-tts`へ導入する。既存`.venv-claip`の
     pinned済みtorch/transformersスタックとの依存衝突を避けるため、橋梁版の「.venv-asr/.venv-claip分離」と同じ思想で
     専用venvを切る
   - 音声モデル`en_US-lessac-medium`（IU X-rayが米語のため）を`piper.download_voices`で`data/tts_voices/`へ取得する

Phase B: 同一ドメイン3モダリティトリプルの構築 (depends on Phase A)
3. 新規`src_claip/synthesize_report_audio.py`を実装する（.venv-tts）
   - `data/processed/image_text/pairs.json`（IU X-ray 1,000件サブサンプル、v0.1.1と同じseed=42で再現）の各`report_text`を
     `PiperVoice.synthesize()`でナレーション音声化する
   - Piperの既定出力22,050Hzを、`encoders_medical.AudioEncoder`が要求する16,000Hz mono PCM16へ`numpy.interp`による
     簡易線形補間でリサンプルする（scipy等の追加依存を避け、.venv-ttsを軽量に保つ）
   - 出力`data/processed/image_text/audio/<image_id>.wav`。1,000件の生成に実測約15分
4. 新規`src_claip/build_triples_single_domain.py`を実装する（.venv-claip, depends on 3）
   - `encoders_medical.ImageTextEncoder`(BiomedCLIP)でimage_emb、新規追加した`AudioEncoder.encode_audio_file()`
     （区間指定不要の全ファイルエンコード、WHISPER_WINDOW_SEC=30sで切り詰め）でaudio_emb、共有`TextEncoder`(multilingual-e5)で
     text_embを同一1,000症例に対して計算し、`image_emb.npy`・`audio_emb.npy`・`text_emb.npy`・`triples.json`を
     `data/processed/image_text/claip_v0.2.0/`へ保存する

Phase C: 3モダリティ対照学習 (depends on Phase B)
5. 新規`src_claip/train_triple_projection.py`を実装する（橋梁版`train_claip_projection.py`の`ProjectionHeads`/
   3ペア加重InfoNCE lossをそのまま移植した`TripleProjectionHeads`）
   - `w_img_txt=w_aud_txt=w_img_aud=1.0`（既定重みなし）、`proj_dim=256`、`epochs=500`、`seed=42`
   - 全バッチ学習（ミニバッチなし）、best-loss checkpointを保存する（v0.1.xと同じ方式）

Phase D: 単一ドメイン検索評価 (depends on Phase C)
6. 新規`src_claip/eval_single_domain_retrieval.py`を実装する
   - クエリ語は`domain_dict/medical_ontology.yaml`のDisease/Symptom/Body-Partカテゴリのうち、IU X-rayレポートコーパスに
     実際に出現する語から頻度順に選定する（v0.1.1の`find_shared_queries()`と異なり、1コーパスのみのチェックで済む
     — 3モダリティとも同一コーパス由来のため）
   - text→image, text→audioの定性的クエリヒット率に加え、**query非依存の大標本指標**として、全1,000トリプルに対する
     image↔audioの closed-set self-retrieval Recall@1/Recall@5（同一症例のaudio_embが正しくtop-1/top-5に入るか）を算出する

Relevant Files
- `0_LogBAK/pre-code_meet_mp4_claip/src_claip/train_claip_projection.py` — `TripleProjectionHeads`/3ペアInfoNCE lossの移植元
- `src_claip/encoders_medical.py` — `AudioEncoder.encode_audio_file()`を追加（区間指定不要の全ファイルエンコード）
- `src_claip/synthesize_report_audio.py` — 新規、Piper TTSによるレポートナレーション音声合成（.venv-tts）
- `src_claip/build_triples_single_domain.py` — 新規、同一症例の(image, audio, text)トリプル構築
- `src_claip/train_triple_projection.py` — 新規、3モダリティ対照学習
- `src_claip/eval_single_domain_retrieval.py` — 新規、単一ドメイン検索評価
- `data/tts_voices/` — Piper音声モデル(en_US-lessac-medium)
- `data/processed/image_text/audio/` — TTS合成音声(1,000件)
- `data/processed/image_text/claip_v0.2.0/` — 3モダリティ埋め込み・射影ヘッド・学習履歴

Verification
- `python src_claip/synthesize_report_audio.py` → `data/processed/image_text/audio/`に1,000件のwavファイル（16kHz mono）が生成されることを確認
- `python src_claip/build_triples_single_domain.py` → `image_emb (1000,512)`, `audio_emb (1000,768)`, `text_emb (1000,768)`が1,000件すべて揃うことを確認（スキップ件数0）
- `python src_claip/train_triple_projection.py --triples-dir data/processed/image_text/claip_v0.2.0 --epochs 500` →
  img-txt/aud-txt/img-aud の3ペアすべてのlossが初期値から低下することを確認（実測: img-txt 6.95→1.18, aud-txt 6.92→0.55, img-aud 6.96→0.88, best total loss 2.55）
- `python src_claip/eval_single_domain_retrieval.py` → `global_image_to_audio_recall@1 >= 0.9`であることを確認（実測0.9970）。
  これは閉集合1,000件に対する大標本指標であり、v0.1.1の少数クエリによる定性チェックより統計的信頼性が高い

Decisions
- 症例レベルで音声×画像が紐付く即座にダウンロード可能な公開データセットは実質存在しないと判断し（PhysioNet egd-cxrはCredentialed
  Health Data Licenseが必須のため不採用）、IU X-rayレポートのTTSナレーション音声合成を採用する。これは「データを探す」という
  ユーザー要求に対する妥協ではなく、「同一症例・同一ドメインのaudio/image/textトリプル」を保証する最も確実な方法であると位置づける
  （合成音声である旨は結果に明記し、実際の人間音声データではない点を隠さない）
- TTSエンジンはCoqui TTS等の重量級ニューラルTTSではなくPiper TTSを採用する: ONNX実行でtorch非依存のため、既存`.venv-claip`の
  pinned依存関係（torch==2.11.0+cu128, transformers>=5,<6等）と衝突するリスクがなく、専用`.venv-tts`に完全分離できる。
  実測で1,000件を約15分で合成でき、品質もニューラル音声として十分自然である
- PriMock57ブランチ（v0.1.x）はv0.2.0のメインCLAIP学習パイプラインから除外する（ユーザー確認済み）。音声ドメインが
  IU X-rayと一致しないため、同一ドメイン3モダリティという本バージョンの核心的な設計目標に合わないため。ただし
  `src/asr_medical.py`等のコード・v0.1.xの実験結果・ドキュメントはアーカイブとして`src/`/`src_claip/`/`docs/`に残す
  （削除しない。将来「同一ドメインの音声診療データ」が入手可能になった場合の参照実装として価値がある）
- `AudioEncoder.encode_audio_file()`を新規追加し、区間指定(`encode_audio_segment`)を前提としないTTS音声全体のエンコードに対応する。
  WHISPER_WINDOW_SEC=30s超の音声（実測1,000件中約10.5%）は先頭30秒のみを使用する（v0.1.x同様の既存の切り詰めポリシーを踏襲）
- image→audio同一症例ヒット率の評価で、TTS音声は報告テキストから生成された決定論的な音声であるため、独立収録された
  人間音声データに比べ「audio-text間の対応学習」自体は易しいタスクである点を結果の解釈に明記する。99%超のRecall@1は
  「CLAIPアーキテクチャが3モダリティ・単一ドメインで正しく機能すること」の実証であり、「実際の自然発話音声理解性能」を
  意味するものではない

Further Considerations
- 実際の人間音声（合成でない）による同一ドメインデータの入手
  推奨: PhysioNet egd-cxrのCredentialed Access取得（CITI研修受講、DUA締結）をv0.3以降の計画として別途進める。
  承認後は本計画のアーキテクチャ（build_triples_single_domain.py等）をほぼそのまま転用できる
- TTS音声が決定論的で「audio-text間の対応学習が易しすぎる」問題への対処
  推奨: 音声合成時にノイズ付加・話速変化・複数話者音声モデルの使用等でaudio-text間のタスク難度を意図的に上げ、
  より頑健なaudio encoderの学習を促す実験をv0.2.1以降で検討する
- PriMock57（非同一ドメイン音声）を再度活用する道
  推奨: 将来、プライマリケア症状×胸部X線所見の両方を含むより広範な医療オントロジー（例: UMLS/SNOMED CTの同義語マッピング）を
  整備すれば、v0.1.xの2ブランチ構成を「概念レベルで接続されたマルチドメインCLAIP」として復活させる道も残しておく
- IU X-ray以外の疾患領域（皮膚科・病理等）への単一ドメイン3モダリティCLAIPの拡張
  推奨: 同様にTTS音声合成を使えば、report/caption付きの他の医用画像データセット（MultiCaRe等）にも同じアーキテクチャを
  横展開できる。v0.3のスコープ候補とする
