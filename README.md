# EEG caffeine input representation CNN

Projekt preizkuša različne vhodne predstavitve EEG signalov za klasifikacijo stanja `Before` proti `After` pri eksperimentu s kofeinom oziroma placebom. Glavni cilj je primerjati modele, ki delajo neposredno na surovih EEG epohah, na ročno izračunanih značilkah ali na kombinaciji obeh predstavitev.

## Kaj projekt vsebuje

- pripravo podatkov iz EEGLAB/MATLAB strukture `ALLEEG`;
- gradnjo dveh `.npz` podatkovnih sklopov:
  - očiščeni EEG: `(epohe, kanali, časovne_tocke)`;
  - značilke: `(epohe, kanali, značilke)`;
- ekstrakcijo časovnih, frekvenčnih in entropijskih značilk;
- učenje in vrednotenje več CNN arhitektur;
- subjektno neodvisno validacijo z `GROUPKFOLD` ali `LOSO`;
- shranjevanje metrik, napovedi, modelov, konfiguracije in grafov v `outputs/`.

Trenutno obdelana sklopa v `data/processed/` vsebujeta 3713 epoh, 32 EEG kanalov in 19 subjektov. Surovi vhod ima obliko `(3713, 32, 900)`, vhod z značilkami pa `(3713, 32, 16)`.

## Struktura projekta

```text
.
├── config.yaml
├── requirements.txt
├── data/
│   ├── raw/                         # surove .mat datoteke, niso namenjene commitu
│   ├── processed/                   # generirani .npz podatkovni sklopi
│   ├── build_raw_dataset.py
│   ├── build_feature_dataset.py
│   └── load_dataset.py
├── notebooks/
│   └── exploration.ipynb
├── src/
│   ├── feature_extraction/          # časovne, frekvenčne in entropijske značilke
│   ├── models/                      # CNN arhitekture
│   ├── train_eval/                  # priprava, učenje, vrednotenje, grafi
│   └── utils/
└── outputs/                         # rezultati poskusov
```

Datoteke `.mat` in `.npz` so v `.gitignore`, ker so velike oziroma generirane. Za ponovitev poskusov mora biti vhodna `.mat` datoteka na poti, nastavljeni v `config.yaml`.

## Namestitev okolja

Priporočen je virtualni Python okoljski imenik. V tem projektu je lokalno okolje ustvarjeno s Python 3.14, vendar koda uporablja standardne knjižnice iz `requirements.txt`.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Če PowerShell blokira aktivacijo okolja, uporabi:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

## Konfiguracija

Glavne nastavitve so v `config.yaml`.

Pomembna polja:

- `experiment.analysis_type` določa, kateri podatki se uporabijo:
  - `caffeine_before_vs_after` uporabi samo skupino `Caffeine`;
  - `placebo_before_vs_after` uporabi samo skupino `Placebo`;
  - `before_vs_after` uporabi vse subjekte.
- `data.data_path` kaže na vhodno `.mat` datoteko.
- `data.raw_dataset_output` in `data.feature_dataset_output` določata izhodni `.npz` datoteki.
- `features` določa nabor značilk in frekvenčne pasove.
- `validation.method` podpira `GROUPKFOLD` in `LOSO`.
- `training` določa epohe, velikost batcha, learning rate, early stopping in napravo.
- `model` vsebuje hiperparametre posameznih arhitektur.
- `outputs` določa mape za modele, metrike in grafe.

## Priprava podatkov

Pred učenjem modelov pripravi oba podatkovna sklopa:

```powershell
python src/train_eval/test_dataset_builders.py
```

Skripta:

- prebere `config.yaml`;
- naloži `ALLEEG` iz `data.data_path`;
- izdela surovi podatkovni sklop;
- izdela podatkovni sklop z značilkami;
- preveri, da se oznake, subjekti, pogoji in skupine med obema sklopoma ujemajo;
- shrani rezultata v `data/processed/raw_dataset.npz` in `data/processed/feature_dataset.npz`.

Oznake so binarne:

- `Before = 0`
- `After = 1`

## Značilke

Za vsak kanal in vsako epoho se lahko izračunajo:

- časovne značilke: `mean`, `std`, `var`, `skewness`, `kurtosis`, `min`, `max`, `hjorth_activity`, `hjorth_mobility`, `hjorth_complexity`;
- frekvenčne značilke: moč v pasovih `delta`, `theta`, `alpha`, `beta`, `gamma`;
- entropijska značilka: `spectral_entropy`.

Pri trenutni konfiguraciji to pomeni 16 značilk na kanal.

## Modeli

Projekt primerja štiri modele:

- `EEGNetLike` v `src/models/cnn_eegnetlike.py` uporablja surove EEG epohe z obliko `(batch, 1, kanali, časovne_tocke)`;
- `CNNFeatures1D` v `src/models/cnn_features_1d.py` uporablja značilke z obliko `(batch, kanali, značilke)`;
- `CNNFeatures2D` v `src/models/cnn_features_2d.py` uporablja značilke kot 2D vhod z obliko `(batch, 1, kanali, značilke)`;
- `CNNHybrid` v `src/models/cnn_hybrid.py` združi surovi EEG vhod in vhod z značilkami.

Vsi modeli uporabljajo PyTorch in vračajo logite za dva razreda.

## Zagon poskusov

Posamezen model lahko zaženeš z eno od skript:

```powershell
python src/train_eval/run_cnn_eegnetlike.py
python src/train_eval/run_cnn_features1d.py
python src/train_eval/run_cnn_features2d.py
python src/train_eval/run_cnn_hybrid.py
```

Vse modele zaporedoma zaženeš z:

```powershell
python src/train_eval/run_all_models.py
```

Ta skripta izmeri čas izvajanja vsakega modela in shrani povzetek v:

```text
outputs/experiment_timings/
```

Če katerikoli model odpove, se zaporedni zagon ustavi, da ne porablja časa za nadaljnje poskuse.

## Analiza rezultatov

Po zagonu modelov lahko primerjaš zadnje rezultate vseh modelov:

```powershell
python src/train_eval/analyze_model_results.py
```

Skripta prebere zadnje mape z rezultati za posamezne modele, izračuna povzetke metrik in shrani primerjalne tabele ter grafe v `outputs/analysis/`.

## Izhodi

Vsak zagon modela ustvari časovno označene mape:

```text
outputs/models/<ime_modela>_<timestamp>/
outputs/results/<ime_modela>_<timestamp>/
outputs/plots/<ime_modela>_<timestamp>/
```

Tipični izhodi so:

- `.pt` kontrolne točke modelov;
- `config_used.yaml`, kopija uporabljene konfiguracije;
- CSV datoteke z metrikami po foldih;
- poročila s povzetki validacije;
- grafi učnih krivulj, matrik zamenjav, ROC krivulj in metrik po foldih;
- `runtime.txt` z informacijami o času izvajanja.

## Validacija

Validacija je subjektno neodvisna: vse epohe istega subjekta ostanejo v istem razdelku.

- `GROUPKFOLD` razdeli subjekte v `n_splits` foldov.
- `LOSO` uporabi Leave-One-Subject-Out pristop, kjer je v vsakem foldu en subjekt testni.

Če je `validation.use_validation_subject` nastavljen na `true`, se validacijski subjekti izberejo samo iz učnega dela folda. To prepreči uhajanje informacij iz testnega subjekta v early stopping.

## Standardizacija

Standardizacija se izvede znotraj vsakega folda:

- za značilke se povprečje in standardni odklon izračunata iz učnega dela;
- za surovi EEG se standardizacija izvede po kanalih;
- validacijski in testni podatki se transformirajo z učnimi statistikami.

S tem se zmanjša tveganje podatkovnega uhajanja med učenjem in vrednotenjem.

## Tipičen potek dela

1. Nastavi `config.yaml`, predvsem `data.data_path`, `experiment.analysis_type` in validacijsko metodo.
2. Pripravi podatke:

```powershell
python src/train_eval/test_dataset_builders.py
```

3. Zaženi en model ali vse modele:

```powershell
python src/train_eval/run_all_models.py
```

4. Primerjaj rezultate:

```powershell
python src/train_eval/analyze_model_results.py
```

5. Preglej izhode v `outputs/results/`, `outputs/plots/` in `outputs/analysis/`.

## Opombe

- Projekt predpostavlja, da `.mat` datoteka vsebuje ključ `ALLEEG`.
- Trenutne skripte nimajo ukaznih argumentov; vse nastavitve se berejo iz `config.yaml`.
- Pri večjih surovih `.mat` in `.npz` datotekah je pričakovana večja poraba RAM-a.
- `outputs/` vsebuje generirane rezultate in se lahko hitro poveča.
