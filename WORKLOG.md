# WORKLOG — skull（顱骨性別分類 pilot，推甄佐證用）

格式：`日期｜決策或發現｜理由一句話｜檔案:行號`。append-only。
真正的專案在 `skulldemo/`；`SkullProject/` 是 1/6 早期版，`SkullProject_Final/` 是空殼。

## 任務卡 2026-09-19：重做 pipeline，兩週內產出可信結果
- 目標：修正前處理與評估方式，重跑分類，拿到一個附 CI、附 baseline 對照的 AUC，整理成推甄可附的乾淨 repo
- 驗收條件：
  1. 前處理輸出：每例同一 spacing、同一 shape、同一 orientation，附 QA 圖與 manifest
  2. 評估：排除 <18 歲，stratified 5-fold CV，AUC + bootstrap 95% CI，旁邊放 majority baseline
  3. 推甄版 repo 不含任何影像／DICOM／checkpoint／病歷號
  4. README 寫清楚 v1 的問題、修正、前後對照
- 非目標：不追求數字好看；不做 segmentation；不改 v1 的舊程式（留作對照）
- 期限：約 2026-10-03（推甄前兩週）

## 紀錄
2026-09-19｜發現 raw_data/ 與 blind_test_nifti/ 的 DICOM 含完整 PHI（姓名、生日、病歷號、開單醫師、accession）｜推甄版絕不能附原始資料；資料本身也該移出 ~/Documents 或去識別化｜raw_data/*/*.dcm header
2026-09-19｜發現 v1 val acc 全是 majority-class 分數（16/21=76.19%、24/31=77.42% 連續多 epoch 不動）｜模型退化成單一輸出，v1 的準確率不能當結果用｜skulldemo/training_history.csv, v7_training_history.csv
2026-09-19｜發現 final_report_cm.png 整體 21/31=67.7%，低於全猜男 23/31=74.2%；女性 4/8｜v1 結果等於 chance｜skulldemo/final_report_cm.png
2026-09-19｜發現 12 個 checkpoint 全是同一架構（18 tensors, 0.6MB），名稱 monai/highres/A100 只是訓練方式不同｜v4→v8 從未換過模型｜skulldemo/checkpoints/*.pth, modules/model.py:2-13
2026-09-19｜發現 v1 前處理各軸獨立 zoom 到 (512,512,32)，原始 matrix 510×662～792×790 不一，spacing 丟失｜形態學訊號被破壞，是最大的方法問題｜skulldemo/modules/data_utils.py:39-41
2026-09-19｜發現 labels_cleaned 含 7 例 <18 歲（最小 8 歲）｜顱骨性別二態性青春期後才明顯，訓練時排除｜skulldemo/labels_cleaned.csv
2026-09-19｜發現 data_utils 讀檔失敗靜默回傳全零張量｜可能有黑影像配真標籤混入訓練，新版要 fail loud｜skulldemo/modules/data_utils.py:32-34
2026-09-19｜發現訓練主程式不在本機（使用者說 notebook 還在，可重跑）｜本機只有 Dataset/model/無 val 的 trainer｜skulldemo/modules/
2026-09-19｜決策：不調參數，改 pipeline；順序＝前處理 → 評估框架 → 模型 → README｜問題在資料與設計，不在 hyperparameter｜本檔任務卡
2026-09-19｜決策：前處理 v2 = RAS → 世界座標重取樣 1×1×5 mm → 骨 bbox 中心裁切 240×240×40 → 存 int16 HU；window 留給 Dataset｜z 保留原 5 mm 不造假切片；bbox p95 229×223×36 所以 240×240×40 幾乎不裁到骨｜skulldemo/preprocess.py:16-18
2026-09-19｜決策：前處理輸出改用匿名 case_id（skull_001–133），對照表只在 preprocessed/manifest.csv｜下游結果與圖表不再出現病歷號｜skulldemo/preprocess.py:66
2026-09-19｜發現 133 例原始 spacing 不一：in-plane 0.26–0.53 mm、z 4.4–7 mm、2 例軸序顛倒（5 mm 在 axis 0）｜證實 v1 各軸獨立 zoom 確實把每個人拉成不同比例｜skulldemo/preprocessed/manifest.csv orig_zooms
2026-09-19｜發現 skull_083、skull_100 是 5 mm 在左右方向（sagittal 厚切），重取樣後 axial 模糊｜與其他 131 例不可比，建議排除（待使用者拍板）｜preprocessed/manifest.csv note 欄
2026-09-19｜發現 skull_064 有大範圍顱面異常＋頭部歪斜，skull_092 bbox 含頭架被裁 5 voxel（非顱骨）｜先留著，064 是否排除待使用者看影像決定｜preprocessed/qa_contact_sheet.png
2026-09-19｜建立 skulldemo/.venv（Python 3.12，uv）｜本機沒有任何有 nibabel/torch 的環境；3.12 對齊舊 pycache｜skulldemo/.venv
2026-09-19｜決策：排除 skull_064（顱面異常）、skull_083、skull_100（sagittal 厚切）；加上 <18 歲 7 例，訓練集 = 123 例｜使用者拍板；異常解剖與不可比的掃描方向不該進形態學分類｜preprocessed/manifest.csv exclude 欄
2026-09-19｜發現訓練 notebook 在 Colab，不在本機｜要嘛下載 .ipynb 放進 skulldemo/，要嘛透過 Google Drive 連接器讀｜—
2026-09-19｜發現 notebook cell[4] 的混淆矩陣與 loss 曲線是手打模擬值（程式註解自述「示範用」「模擬出的預測結果」）｜該 cell 產出的 confusion_matrix.png / learning_curves.png 絕不可用於推甄；硬碟上 final_report_cm.png 來自 cell[8] 真實 val 預測，可用｜ＨＥＡＤMonAI.ipynb cell[4]
2026-09-19｜發現 v1「盲測」取的是 nifti_data 最後 15 筆（在訓練池內），blind_test_nifti/ 是 DICOM 從未被讀到｜v1 沒有任何真正 held-out 的結果｜ＨＥＡＤMonAI.ipynb cell[5][7][18][19]
2026-09-19｜發現 v6→v7→v8→girl1→A100 逐版載入前版權重但 train/val 每版重切｜跨版本 val 洩漏，v1 的 val acc 全部不可信｜ＨＥＡＤMonAI.ipynb cell[1][3][6][17] OLD_CKPT
2026-09-19｜發現 v1 用 ScaleIntensityd 逐例 min-max，6 例含金屬 HU max 5k–17.5k｜同一組織在不同人數值差 8 倍；v2 改固定 HU window｜ＨＥＡＤMonAI.ipynb cell[3] transforms
2026-09-19｜未確認：v8_girl1 訓練時印出男 112／女 23，與現行標籤檔 75/58 不符；標籤檔 1/26 修改晚於 1/16 訓練｜當時所用標籤不可考，README 只能寫「無法重現」｜ＨＥＡＤMonAI.ipynb cell[6] output
2026-09-19｜發現 v1 A100 版只對有 seg 的 10 例套顱骨 mask，其餘 148 例不套｜同一模型內前處理不一致｜ＨＥＡＤMonAI.ipynb cell[17] ApplySkullMaskd
2026-09-19｜決策：v2 訓練 = DenseNet121-3D (MONAI, 11.2M) + 固定骨窗 HU[-500,1300] + 增強只做左右翻轉與 ≤8/8/2 voxel 平移｜小樣本用中型現成網路；固定 window 取代逐例 min-max；增強不做插值以免 CPU 成瓶頸｜skulldemo/train_cv.py:22-23,42-48
2026-09-19｜決策：評估 = StratifiedKFold 5 × 固定 30 epoch，取最後 epoch 的 OOF 預測合併算 AUC + bootstrap 95% CI，不用 val 挑 epoch｜沒有獨立 test set，用 val 選模型會灌水；固定 epoch 是誠實作法，README 要寫明｜skulldemo/train_cv.py:63-70,95-99
2026-09-19｜發現本機 M4 Pro MPS 可跑 3D DenseNet，約 0.5 s/例；完整 5-fold 估 2–2.5 h｜本機可過夜跑，Colab A100 可平行跑對照｜skulldemo/results/smoke
2026-09-19｜smoke test 通過（12 例 2 fold 2 epoch，24 s），啟動正式 run：results/v2_run1｜—｜skulldemo/train_cv.py
2026-09-19｜建立推甄 repo 骨架 skull-sex-cnn/：README.md 草稿（結果欄【待填】）+ requirements.txt｜README 主軸 = v1→v2 的問題清單與修正，不是數字；程式、結果、v1 對照檔待訓練完再複製進去｜skull-sex-cnn/README.md
2026-09-19｜發現納入 123 例男女年齡分布相同（中位數皆 68，Mann-Whitney p=0.92）｜年齡不是混淆因子，README §5 據此改寫｜skull-sex-cnn/README.md §5
2026-09-19｜待使用者決定：README §4.1「模擬混淆矩陣」那一列要不要保留｜誠實但可能被誤讀為造假；cell 註解明寫「示範用」｜skull-sex-cnn/README.md §4.1
2026-09-19｜v2_run1 fold 0 到 ep 29：val AUC 0.98、acc 0.92（25 例）｜遠高於預期，先懷疑捷徑（頭大＝男）再高興；等其餘 4 折｜skulldemo/results/v2_run1/train.log
2026-09-19｜新增 size_baseline.py：只用顱骨大小（骨體素數 + bbox xyz）做 LR，同 fold 切法｜檢查 CNN 有多少只是在量頭大小；README 需要這個對照｜skulldemo/size_baseline.py
2026-09-19｜尺寸 baseline 結果：骨體素數單獨 AUC 0.61 (0.51–0.71)；bbox xyz AUC 0.77 (0.69–0.85)；四項合併 0.77｜頭大小解釋一部分但不到 0.98；bbox_z 是掃描範圍非純解剖，解讀要小心｜skulldemo/results/size_baseline/metrics.json
2026-09-19｜README 加入 §3.1 尺寸 baseline 表（AUC 0.61/0.77/0.77）與 bbox_z 的解讀提醒｜回答「是不是頭大就男」；CNN 要明顯高於 0.77 才算學到形態｜skull-sex-cnn/README.md §3.1
2026-09-20｜v2_run1 fold 1 完成：val AUC 0.981、acc 0.92（25 例），與 fold 0 一致｜兩折同高，運氣解釋不成立；仍待 3 折｜skulldemo/results/v2_run1/train.log
2026-09-20｜v2_run1 fold 2 至 ep 27：AUC 1.000｜三折 0.98–1.0，需排除捷徑｜skulldemo/results/v2_run1/train.log
2026-09-20｜檢查 metadata 捷徑：spacing/矩陣/FOV/日期/病歷號前綴單一 AUC ≤ 0.62，切片數 0.71｜不是掃描協定造成；train_cv.py 重讀無 train/val 重疊｜本 session 臨時腳本
2026-09-20｜發現骨窗 [-500,1300] 仍看得到軟組織、頭髮（0.22–0.28）與耳環（飽和）｜目前結果是「頭部 CT 判性別」而非「顱骨判性別」；README 措辭要對應｜skulldemo/train_cv.py:22
2026-09-20｜計畫：run1 完成後 (1) 標籤打亂 sanity check 1 折 10 epoch，預期 AUC≈0.5；(2) 純骨 window [300,1300] 重跑 5 折，待使用者拍板｜區分頭髮／軟組織與顱骨的貢獻｜—
2026-09-20｜v2_run1 fold 2 完成：val AUC 1.000、acc 1.000（25 例）｜三折 0.98/0.98/1.00｜skulldemo/results/v2_run1/train.log
2026-09-20｜v2_run1 fold 3 完成：val AUC 0.993、acc 0.917（24 例）｜四折 0.987/0.981/1.000/0.993｜skulldemo/results/v2_run1/train.log
2026-09-20｜v2_run1 完成：OOF AUC 0.987 (95% CI 0.971–0.997)，各折 0.991±0.007，acc 0.935，sens(女) 0.889，spec(男) 0.971，majority 0.561；錯 8/123（女→男 6、男→女 2）｜遠高於尺寸 baseline 0.77；在確認非洩漏／非捷徑前不寫進 README 結論｜skulldemo/results/v2_run1/metrics.json
2026-09-20｜啟動 sanity_shuffle.py：標籤打亂、1 折 10 epoch｜預期 val AUC≈0.5；若仍高則有洩漏 bug｜skulldemo/results/sanity_shuffle/log.txt
2026-09-20｜發現 [-500,1300] 骨窗下頭皮、頭髮暈、臉部軟組織、頭架全部可見；判錯的女性頭頂切面有厚軟組織暈｜run1 的 0.987 是「頭部 CT 判性別」，顱骨貢獻未知｜scratchpad/window_view.png
2026-09-20｜發現 >300 HU 最大連通元件即顱骨（430–830 cm³）；頭架是分離元件（HU≈326, 28 cm³）；耳環等 <2 cm³｜純骨定義可用「最大連通元件」一句話說清楚｜skulldemo/train_cv.py:28-31
2026-09-20｜決策：train_cv.py 加 --bone_only（最大連通骨元件 + window [300,1300]）；sanity 跑完自動啟動 results/v2_bone_only｜區分顱骨 vs 軟組織／頭髮的貢獻，這是 README 的核心對照｜skulldemo/train_cv.py:109-112
2026-09-20｜sanity_shuffle ep1：val AUC 0.565（打亂標籤）｜初步正常，等 10 epoch 全部｜skulldemo/results/sanity_shuffle/log.txt
2026-09-20｜使用者指示：標籤測試完先暫停，不自動啟動純骨版｜已取消自動接續；純骨版待指示｜—
2026-09-20｜sanity_shuffle 完成：打亂標籤 10 epoch val AUC 0.31–0.57，最後 0.43｜無洩漏；同設定真標籤同一折 ep10 為 0.94，差距明確｜skulldemo/results/sanity_shuffle/log.txt
2026-09-20｜決策：純骨版以降溫模式跑：先休 20 min，之後每 epoch 後休 20 min（--cooldown 1200）｜使用者要求讓電腦降溫；150 epoch 估 53 h，約 9/22 晚完成｜skulldemo/train_cv.py --cooldown, results/v2_bone_only/train.log
2026-09-20｜更正：冷卻改為每「折」之間 20 min（非每 epoch）；初始 20 min 冷卻沿用｜使用者更正；總時程 ≈ 20 min + 5×32 min + 4×20 min ≈ 4.3 h，約 06:20 完成｜skulldemo/train_cv.py --cooldown（fold 間）
2026-09-20｜v2_bone_only fold 0 完成：val AUC 0.968、acc 0.92（run1 同折 0.987/0.92）｜純骨略低但仍 >0.95，頭髮／軟組織不是主要訊號｜skulldemo/results/v2_bone_only/train.log
2026-09-20｜v2_bone_only fold 1 完成：val AUC 0.968、acc 0.88（run1 同折 0.981/0.92）｜兩折純骨皆 0.968｜skulldemo/results/v2_bone_only/train.log
2026-09-20｜v2_bone_only fold 2 完成：val AUC 1.000、acc 0.96（run1 同折 1.000/1.00）｜三折純骨 0.968/0.968/1.000｜skulldemo/results/v2_bone_only/train.log
2026-09-20｜v2_bone_only fold 3 完成：val AUC 0.936、acc 0.875（run1 同折 0.993/0.917）｜四折純骨 0.968/0.968/1.000/0.936，此折差距較大 (−0.057)｜skulldemo/results/v2_bone_only/train.log
2026-09-20｜v2_bone_only 完成：OOF AUC 0.974 (0.945–0.995)，各折 0.970±0.021，acc 0.919，sens(女) 0.926，spec(男) 0.913；錯 10/123｜純骨比全頭部低 0.013，CI 大幅重疊；訊號主要在顱骨，頭髮／軟組織貢獻很小｜skulldemo/results/v2_bone_only/metrics.json
2026-09-20｜兩版判錯交集 6 例（029/046/088/099/114/121）；只有純骨錯 4 例（015/057/086/113）｜錯的病例大致相同，兩版看的是同一種訊號｜results/*/oof_predictions.csv
2026-09-20｜結論鏈完成：非過擬合（val 好）、非洩漏（打亂標籤 0.43）、非 metadata 捷徑（≤0.62）、非頭髮／軟組織捷徑（純骨 0.974）；尺寸只解釋 0.77｜可以寫進 README 結果；剩下的開放問題是模型看顱骨哪裡（可解釋性）｜—
2026-09-20｜決策：可解釋性用 Grad-CAM，層取 features.denseblock2（30×30×5）｜denseblock4 只有 7×7×1，z 無解析度；denseblock2 約 8 mm 面內、40 mm 縱向｜skulldemo/explain.py:20
2026-09-20｜決策：explain.py 重訓純骨 fold 0 並存權重到 results/explain_fold0/model.pt（不進推甄 repo）｜train_cv 不存權重；CAM 需要模型｜skulldemo/explain.py
2026-09-20｜run_fold 改回傳 (p, model)｜explain.py 需要模型物件｜skulldemo/train_cv.py:87
2026-09-20｜Grad-CAM (denseblock2) 結果作廢：男性 57–74%、女性 ~5% 體素飽和在 1.0 且集中在空氣區，顱骨反而低｜3D 大面積 padding 下空氣區特徵由 bias 主導形成常數底，與類別無關；cam_*.png 不可用於解讀｜skulldemo/results/explain_fold0/cams_fold0_val.npy
2026-09-20｜決策：可解釋性改用 occlusion sensitivity（遮一塊顱骨換成空氣，量 logit 變化）｜意義明確、無 ReLU／正規化的曖昧；沿用 explain_fold0/model.pt 與同一批 held-out 病例｜skulldemo/occlusion.py
2026-09-20｜occlusion.py 啟動：遮罩 24×24×40 mm、步長一半、只遮含骨位置（~1,300/例）、25 例 held-out｜每例 76 s，全程約 32 min｜skulldemo/results/occlusion_fold0/log.txt
2026-09-20｜occlusion 完成（fold 0 held-out 25 例，24/25 判對）：女性證據集中在中上顱頂（vault+mid 佔 0.98，顱底 0.02）；男性證據集中在顱底層（前方眶上／額區 0.24、後方枕／乳突區 0.12）與頭頂 0.22，且顱頂對男性是反向證據（遮掉反而更像男）｜位置與經典二態性特徵（眉弓、乳突、枕外隆凸）一致，但解析度 24×24×40 mm 無法指認具體結構；解剖解讀待使用者確認｜skulldemo/results/occlusion_fold0/occ_summary.json, occ_mean.png
2026-09-20｜occlusion 每例最重要遮罩：男性多在 z=12（顱底層）前方 y≈180–192 或後方 y≈48–84；女性多在 z=24–28（上顱頂）前方 y≈156–180｜個案與群體圖一致｜occ_summary.json cases[]
2026-09-20｜README §3 填完：主結果表（純顱骨／全頭部／尺寸／majority）、驗證鏈四項、occlusion 九宮格與解讀、Grad-CAM 作廢說明；§2/§5/§6/§7 同步｜所有數字來自 results/*/metrics.json 與 occ_summary.json｜skull-sex-cnn/README.md
2026-09-20｜推甄 repo skull-sex-cnn/ 組裝完成（31 檔、3.5 MB）：6 支程式、兩版結果、尺寸 baseline、sanity 紀錄、occlusion 圖表、v1 對照、WORKLOG 快照、.gitignore｜全文掃描無病歷號／機構名／人名／原始檔名；PNG 只含 matplotlib Software 標籤｜skull-sex-cnn/
2026-09-20｜待使用者：README 四處【填】（作者、機構、IRB×2）、一處【請確認解剖解讀】、一處【是否保留模擬圖那列】；git init 與是否公開由使用者決定｜—｜skull-sex-cnn/README.md:5,126,151,253
2026-09-20｜啟動 seed 重複：純骨版 seed 43 → 冷卻 20 min → seed 44，每折間冷卻 20 min｜主結果要報多 seed 的 mean±SD；兩 run 約 8 h｜skulldemo/results/v2_bone_only_seed43/, seed44/
2026-09-20｜seed43 fold 0 完成：val AUC 0.994、acc 0.96｜—｜skulldemo/results/v2_bone_only_seed43/train.log
2026-09-20｜seed43 fold 1 完成：val AUC 0.981、acc 0.92｜—｜skulldemo/results/v2_bone_only_seed43/train.log
2026-09-20｜使用者指示：只跑 seed 43，跑完暫停；已移除 seed 44 串接｜—｜skulldemo/results/v2_bone_only_seed43/
2026-09-20｜seed43 fold 2 完成：val AUC 1.000、acc 0.96｜三折 0.994/0.981/1.000｜skulldemo/results/v2_bone_only_seed43/train.log
2026-09-20｜seed43 fold 3 完成：val AUC 0.914、acc 0.833｜四折 0.994/0.981/1.000/0.914；跟 seed42 一樣有一折偏低｜skulldemo/results/v2_bone_only_seed43/train.log
2026-09-20｜seed43 純骨版完成：OOF AUC 0.976 (0.953–0.993)，各折 0.994/0.981/1.000/0.914/0.958，acc 0.902；兩 seed 平均 0.975±0.001，p(F) 相關 0.926｜主結果穩定；README §3.1/3.5/6/7 已更新，結果已複製進 repo｜skulldemo/results/v2_bone_only_seed43/metrics.json
2026-09-20｜發現 5 例三個 run（全頭部、純骨 s42、純骨 s43）都判錯：skull_046/088/099/114/121｜錯誤分析（待使用者看影像）從這五例開始｜results/*/oof_predictions.csv
2026-09-20｜explain.py / occlusion.py 加 --fold；驗證 fold 0 名單與既有 occlusion_fold0 一致｜補滿五折需要逐折重訓與遮罩｜skulldemo/explain.py:33,40, occlusion.py --fold
2026-09-20｜啟動 occlusion 鏈（使用者出門斷網）：fold 1 訓練→冷卻 20→遮罩→冷卻 20→fold 2 訓練→冷卻 20→遮罩→停；進度在 results/occlusion_chain.log｜使用者要求先做兩折，回來再續｜skulldemo/results/occlusion_chain.log
2026-09-20｜fold 2 訓練完成（val AUC 見 explain_fold2/log.txt），依使用者要求暫停鏈；occlusion fold 2 待啟動｜使用者出門合蓋｜skulldemo/results/occlusion_chain.log
2026-09-20｜使用者回來，啟動 occlusion fold 2｜—｜skulldemo/results/occlusion_fold2/
2026-09-20｜使用者：散熱已處理，取消冷卻；排入 fold 3、4 訓練＋occlusion 連跑，fold 2 occlusion 結束後自動接續｜五折補齊約 2.2 h｜skulldemo/results/occlusion_chain.log
2026-09-20｜五折 occlusion 完成並合併（123 例，115 判對）：女性顱底 0.01／中層 0.60／顱頂 0.40；男性顱底 0.50（前 0.22、後 0.12）／中層 0.12／頭頂 0.38；關鍵格子五折一致｜fold 0 的結論在全部五折重現；README §3.4 改為五折版，§5「只做一折」限制移除｜skulldemo/results/occlusion_pooled/occ_pooled_summary.json
2026-09-20｜新增 occlusion_pool.py；repo 補入 occlusion_fold1–4 與 occlusion_pooled｜—｜skull-sex-cnn/occlusion_pool.py, results/occlusion_*
2026-09-20｜各折重訓模型 val AUC：fold0 0.981、fold1 0.994、fold2 1.000、fold3 0.957、fold4 0.965（與主結果的模型非同一組權重）｜README §5 已註明｜skulldemo/results/explain_fold*/log.txt
2026-09-20｜決策：專案定位為 MVP，不做人類盲判｜使用者拍板；推甄佐證的主體改為「經驗敘述 + repo」｜—
2026-09-20｜寫推甄用經歷草稿 application_mvp_writeup.md（第一人稱、約一頁）；六處措辭已校回證據強度（val 男性人數為推論、金屬壓縮倍率、metadata 例外、occlusion 比例、檢查耗時）｜草稿數字全部對應 results/*；三處【填】一處【請確認】｜application_mvp_writeup.md
2026-09-20｜application_mvp_writeup.md 去 AI 味＋降技術深度，19 處全套用；原稿備份 ~/.claude/backups/2026-09-20/｜使用者要求「不要寫得太深」；數字與【填】未動｜application_mvp_writeup.md

## 任務卡 2026-09-20：年齡迴歸 pilot（skullage/）
- 目標：用同一批 123 例（排除 <18、064、083、100）與同一套前處理輸出，訓練 DenseNet121-3D 迴歸年齡，看有沒有超過「猜平均」的訊號
- 驗收條件：
  1. skullage/train_age_cv.py 5-fold CV 跑完，輸出 metrics.json（MAE + bootstrap CI、RMSE、r、各年齡層 MAE／偏差）、oof_predictions.csv、summary.png
  2. 旁邊放兩個 baseline：猜訓練折平均年齡；簡單特徵（骨量、bbox、骨平均 HU）ridge regression
  3. 不動 skulldemo/ 任何程式；資料直接讀 ../skulldemo/preprocessed/，不複製
- 非目標：不進推甄 repo（先看結果再說）；不含 <18 歲；不調參
2026-09-20｜決策：年齡模型不放 <18 歲 7 例，維持 123 例與性別模型可比｜使用者拍板；小兒顱骨是另一種生物過程，且只有 7 例｜skullage/
2026-09-20｜決策：第一輪用全頭部窗 [-500,1300]，不用 bone_only｜bone_only 只留最大連通骨元件，會把顱內鈣化（松果體、頸動脈虹吸部）一併去掉，那可能是年齡的主要線索；bone_only 留作第二輪對照｜skullage/train_age_cv.py
2026-09-20｜年齡分布：18–96，mean 65.5、SD 18.9，40 歲以下只 12 例；猜平均 MAE 15.8 歲｜年齡模型要壓過這個數字才算有訊號；年輕人預期會被系統性猜老｜skulldemo/preprocessed/manifest.csv age 欄
2026-09-20｜skullage/train_age_cv.py 完成（複製 train_cv.py 改迴歸：age 標籤、訓練折 z-score、L1 loss、年齡五分位分折、MAE/RMSE/r + CI、各年齡層偏差）；smoke 通過｜與性別模型同網路同設定，只換任務；skulldemo/ 未動｜skullage/train_age_cv.py
2026-09-20｜簡單特徵 baseline（ridge，同折）：骨平均 HU 單獨 MAE 14.38 (12.45–16.35)、r +0.31；尺寸四項 MAE 16.07、r −0.06；猜平均 15.83｜骨密度代理（骨平均 HU 與年齡 r −0.33）有弱訊號，顱骨大小對年齡沒用；CNN 要壓過 14.4 才算學到尺寸／密度以外的東西｜skullage/results/feature_baseline/metrics.json
2026-09-20｜啟動年齡 CNN 正式 run（全頭部窗、123 例、5 折 30 epoch、seed 42，pid 48554）｜每 epoch 64 s，預計 2.7 h｜skullage/results/age_fullhead/train.log
2026-09-20｜年齡 CNN fold 0 完成：val MAE 10.42（ep 23 最低 9.64，後段 9.6–10.7 平穩）｜明顯壓過骨 HU baseline 14.4 與猜平均 15.8，有訊號；train loss 0.40 SD ≈ 7.6 歲，過擬合幅度可接受｜skullage/results/age_fullhead/train.log
2026-09-20｜排入 bone_only 第二輪：chain_bone_only.sh 等第一輪 pid 48554 結束後自動啟動 --bone_only → results/age_bone_only/；使用者要求跑完通知｜第二輪看純顱骨形態（去掉顱內鈣化）還剩多少年齡訊號｜skullage/chain_bone_only.sh, results/chain.log
2026-09-20｜年齡 CNN fold 1 完成：val MAE 10.98｜兩折 10.42/10.98，一致｜skullage/results/age_fullhead/train.log
2026-09-20｜年齡 CNN fold 2 完成：val MAE 11.00｜三折 10.42/10.98/11.00｜skullage/results/age_fullhead/train.log
2026-09-20｜年齡 CNN fold 3 完成：val MAE 10.20｜四折 10.42/10.98/11.00/10.20｜skullage/results/age_fullhead/train.log
2026-09-20｜年齡 CNN 第一輪（全頭部窗）完成：OOF MAE 11.10 (95% CI 9.58–12.71)、RMSE 14.0、r 0.68；各折 10.42/10.98/11.00/10.20/12.91｜壓過猜平均 15.83 與骨 HU ridge 14.38；有骨密度以外的訊號｜skullage/results/age_fullhead/metrics.json
2026-09-20｜年齡層偏差呈迴歸均值：18–39 歲（12 例）被猜老 +18.1、MAE 18.6；40–69 歲 +3～+7；80+（37 例）被猜年輕 −10.2｜年輕人少、模型往 65 歲收縮；解讀時要註明年齡分布偏老、極端年齡不可信｜skullage/results/age_fullhead/metrics.json by_age_group
2026-09-20｜fold 4 偏差（12.91 vs 其他 10.2–11.0）；學習曲線前 15 epoch 各折震盪大（MAE 20–40），20 epoch 後才穩｜30 epoch 對迴歸剛好夠；不調參，先看 bone_only｜skullage/results/age_fullhead/summary.png
2026-09-20｜bone_only 第二輪已由 chain 自動啟動（21:31）｜預計 2.7 h｜skullage/results/age_bone_only/train.log
2026-09-20｜skull-sex-cnn/ git init + 首次 commit（50 檔，無影像／標籤／權重）；作者設為使用者本名與 email（repo-local）｜使用者要求「可以 git 了，注意敏感資料」；commit 前後各掃一次｜skull-sex-cnn/.git
2026-09-20｜bone_only fold 0 完成：val MAE 12.88（全頭部同折 10.42）；fold 1 ep 29 在 12.84（全頭部 10.98）｜去掉顱內鈣化／軟組織後退約 2 歲，方向如預期，等五折｜skullage/results/age_bone_only/train.log
2026-09-20｜bone_only fold 1 完成：val MAE 12.76｜兩折 12.88/12.76 vs 全頭部 10.42/10.98｜skullage/results/age_bone_only/train.log
2026-09-20｜推送到 GitHub 私人 repo https://github.com/alwayscrush0124/skull-sex（main, 25590d5）；推甄稿 repo 連結已填｜使用者提供網址即授權推送；repo 為 PRIVATE｜skull-sex-cnn/.git, application_mvp_writeup.md
2026-09-20｜bone_only fold 2 完成：val MAE 13.27｜三折 12.88/12.76/13.27 vs 全頭部 10.42/10.98/11.00，穩定差 2–2.5 歲｜skullage/results/age_bone_only/train.log
2026-09-20｜bone_only fold 3 完成：val MAE 12.39｜四折 12.88/12.76/13.27/12.39 vs 全頭部 10.42/10.98/11.00/10.20｜skullage/results/age_bone_only/train.log
2026-09-21｜年齡 CNN 第二輪（bone_only）完成：OOF MAE 12.66 (95% CI 10.90–14.45)、RMSE 16.1、r 0.57；各折 12.88/12.76/13.27/12.39/11.95｜比全頭部版 11.10 差 1.6 歲，CI 大幅重疊；純顱骨仍壓過猜平均 15.83 與骨 HU ridge 14.38，但軟組織／顱內鈣化對年齡的貢獻比對性別大（性別兩版只差 0.013 AUC）｜skullage/results/age_bone_only/metrics.json
2026-09-21｜bone_only 各年齡層偏差比全頭部更極端：18–39 +19.2、40–49 +13.5、80+ −12.5｜純顱骨版更往均值收縮；兩版都不能用在 40 歲以下｜skullage/results/age_bone_only/metrics.json by_age_group
2026-09-21｜任務卡驗收：1. 兩輪 metrics.json/oof_predictions.csv/summary.png 齊全 ✓ 2. 猜平均與 ridge baseline 齊全 ✓ 3. skulldemo/ 未動、資料未複製 ✓｜年齡 pilot 完成；是否進推甄 repo 待使用者決定｜skullage/results/
2026-09-21｜使用者釐清用途：年齡 MVP 是包裝進推甄 AI 研究所 CV，讓電資老師看到有在動手做；不再跑 seed、不做可解釋性｜寫 skullage/README.md（一頁技術說明，格式同性別 README，所有數字來自 results/*/metrics.json）；下一步併進 skull-sex-cnn/age/ 與 CV 條目文字待使用者同意｜skullage/README.md
2026-09-21｜使用者決定：AI 協作明寫（非電資本科，正是申請理由）；repo 維持一個，年齡併入；CV 文字傳給「推甄資料結構規劃」session｜—｜—
2026-09-21｜年齡延伸併入推甄 repo：skull-sex-cnn/age/（兩支程式、README、三個 results 各 metrics/history/summary）；主 README 改標題、加「開發方式」一行、§5 指向、§7 檔案清單、新增 §9；原 README 備份 ~/.claude/backups/2026-09-21/｜repo 不收年齡 oof_predictions.csv（含病例層級年齡，與 repo 既有排除政策一致）與 train.log（含本機路徑）；PHI 掃描 clean｜skull-sex-cnn/README.md:1,5-7,181,257-262,270-287
2026-09-21｜CV 條列版＋段落版已傳給「推甄資料結構規劃」session，附數字依據與三個決定｜git commit / push 由使用者決定｜—

## 結案摘要 2026-09-21（session 結束）
任務卡驗收（2026-09-19 開的卡）：
1. 前處理輸出同 spacing／shape／orientation + QA + manifest → ✅ skulldemo/preprocessed/（133 例，240×240×40，1×1×5 mm，RAS）、qa_contact_sheet.png、manifest.csv
2. 排除 <18、5-fold、AUC+CI、majority baseline → ✅ results/v2_bone_only/metrics.json（0.974, CI 0.945–0.995；seed43 0.976）；另有尺寸 baseline 0.77、標籤打亂 0.43、五折 occlusion
3. 推甄 repo 無影像／DICOM／權重／病歷號 → ✅ https://github.com/alwayscrush0124/skull-sex（PRIVATE，2 commits，commit 前後掃描皆乾淨）
4. README 寫清楚 v1 問題／修正／對照 → ✅ skull-sex-cnn/README.md §4（使用者已填作者、機構；IRB 行與模擬圖列由使用者決定刪除）
額外產出：application_mvp_writeup.md（推甄經歷稿，已去 AI 味、降深度，repo 連結已填；第 9 行句子使用者尚未順完）

待辦（下次接續）：
- 使用者：順推甄稿第 9 行；推甄前決定 repo 是否改 public（README 有 NTUH 字樣、無 IRB 聲明）
- 可選：explain.py 移除作廢 Grad-CAM 段、加 LICENSE、README 英文摘要
- 資料安全：skulldemo/raw_data、blind_test_nifti、nifti_data、seg 含病歷號與 DICOM PHI，仍在 ~/Documents；推甄用不到，應移至加密碟或刪除（需使用者決定）
- 不做：人類盲判（使用者定位為 MVP）

## 補記 2026-09-23：年齡迴歸延伸（2026-09-21 於另一 session 完成，結案摘要當時漏記）
2026-09-21｜完成年齡迴歸延伸 age/：同 123 例、同前處理、同 DenseNet121-3D，標籤換年齡、BCE 換 L1、分折改依年齡五分位 stratified｜驗證同一套 pipeline 換任務的可行性；MVP 的加分題｜skull-sex-cnn/age/train_age_cv.py, age/README.md
2026-09-21｜年齡結果：全頭部 OOF MAE 11.10 歲（95% CI 9.58–12.71），r=0.68；純顱骨 12.66（10.90–14.45），r=0.57；猜平均 baseline 15.83｜兩版都贏 baseline，但幅度遠小於性別任務｜age/results/age_fullhead/metrics.json, age_bone_only/metrics.json
2026-09-21｜發現年齡任務與性別任務相反：全頭部優於純顱骨 1.56 歲，CI 部分重疊｜性別的訊號在顱骨，年齡的訊號有一部分在顱骨以外（軟組織／血管鈣化？未驗證）｜age/results/*/metrics.json
2026-09-21｜特徵 baseline：骨平均 HU 單獨 MAE 14.38，優於尺寸 16.07；原始特徵與年齡相關 骨平均 HU −0.33、骨體素數 −0.13、bbox 三軸 ≤0.06｜年齡訊號來自骨密度而非大小，與性別任務（尺寸 AUC 0.77）相反｜age/results/feature_baseline/metrics.json
2026-09-21｜發現年輕組誤差最大：18–39 歲 n=12，全頭部 MAE 18.6、bias +18.1（系統性高估）｜樣本偏老（中位 68，80+ 佔 37 例）導致迴歸向均值；README 需註明此限制｜age/results/age_fullhead/metrics.json by_age_group
2026-09-23｜更正：09-21 結案摘要的「額外產出」漏列 age/ 延伸（該 commit 6b61b72 早於摘要 11 小時）｜當時只查 git log 最新一筆未看全史；本節補記｜WORKLOG.md 結案摘要

待辦更新（取代前節待辦的第 2 項）：
- age/README 已完整涵蓋年輕組高估（§3.2 分層表、§3.3 迴歸均值、§4 限制），無需補寫
- 待辦不變：順推甄稿第 9 行、決定 repo 是否 public、處理 skulldemo/ 的 PHI 資料
- 推甄稿 application_mvp_writeup.md 目前只寫性別任務，未提 age/ 延伸（是否加入由使用者決定）
