# EEG caffeine input representation CNN

Projekt primerja štiri CNN-arhitekture za binarno razvrščanje EEG-epoh Before in After v eksperimentu s kofeinom oziroma placebom: surovi EEG, 1D-značilke, 2D-značilke in hibridni vhod. Udeleženci so med učenjem, validacijo in testiranjem ločeni.

Aktivna konfiguracija je v config.yaml označena z različico 1.0.1. Spremembe od začetne različice so v [CHANGES.md](CHANGES.md): enotne mape poskusov, gnezdena Optuna optimizacija, referenčni klasifikatorji, statistični testi in razširjena EDA.

## Zbirki in struktura

Trenutni zbirki v data/processed/ vsebujeta 3.713 epoh, 32 kanalov in 19 udeležencev:

~~~
raw_dataset.npz:     (3713, 32, 900)
feature_dataset.npz: (3713, 32, 13)
~~~

Končna analiza se glede na experiment.analysis_type omeji na Caffeine, Placebo ali obe skupini; trenutno je izbrana skupina Caffeine.

~~~
.
├── CHANGES.md
├── config.yaml
├── requirements.txt
├── data/
│   ├── raw/                         # vhodne .mat datoteke
│   ├── processed/                   # ustvarjeni .npz zbirki
│   ├── build_raw_dataset.py
│   ├── build_feature_dataset.py
│   └── load_dataset.py
├── notebooks/
│   ├── exploration.ipynb            # EDA
│   └── eda_figures/
├── src/
│   ├── feature_extraction/
│   ├── models/
│   ├── train_eval/
│   └── utils/
└── outputs/
    ├── experiments/
    ├── hyperparameter_optimization/
    └── archive/
~~~

Velike in ustvarjene datoteke .mat/.npz so v .gitignore. Vhodna datoteka mora na data.data_path vsebovati ključ ALLEEG.

## Namestitev

Lokalno okolje uporablja Python 3.14. Namesti zahteve, tudi Optuna 4.x:

~~~powershell
python -m venv .venv
./.venv/Scripts/Activate.ps1
pip install -r requirements.txt
~~~

Če PowerShell blokira aktivacijo:

~~~powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
~~~

## Konfiguracija in priprava podatkov

config.yaml določa izbor skupine, poti podatkov, frekvenco vzorčenja, skupine značilk, validacijo, Optuna nastavitve, učenje, arhitekture in izhode. Trenutna privzeta validacija je GROUPKFOLD s petimi foldi in dvema validacijskima udeležencema. Pri use_validation_subject: true se validacijski udeleženci izberejo samo iz učnega dela.

~~~powershell
python src/train_eval/test_dataset_builders.py
~~~

Skripta prebere ALLEEG, uporabi include_conditions in exclude_subjects, zgradi zbirki, preveri ujemanje oznak, udeležencev, pogojev in skupin ter ju shrani v data/processed/. Razreda sta Before = 0 in After = 1.

## Značilke

Trenutni vhod z značilkami ima 13 vrednosti na kanal:

- časovne: mean, std, skewness, kurtosis, min, max, hjorth_mobility, hjorth_complexity;
- frekvenčne: log10 absolutna moč v theta (4–8 Hz), alpha (8–13 Hz), beta (13–30 Hz) in gamma (30–45 Hz);
- entropijska: spectral_entropy.

Welchova ocena za moč in entropijo uporablja celotno epoho: welch_nperseg: 900 pri 600 Hz. Delta pas je odstranjen, ker 1,5-sekundne epohe nimajo zanesljive ločljivosti. Skupine značilk je mogoče v konfiguraciji izključiti.

## Modeli in preprečevanje uhajanja podatkov

| Model | Datoteka | Vhod |
| --- | --- | --- |
| EEGNetLike | src/models/cnn_eegnetlike.py | (batch, 1, kanali, časovne_točke) |
| CNNFeatures1D | src/models/cnn_features_1d.py | (batch, kanali, značilke) |
| CNNFeatures2D | src/models/cnn_features_2d.py | (batch, 1, kanali, značilke) |
| CNNHybrid | src/models/cnn_hybrid.py | surovi EEG in značilke |

Modeli vrnejo logite za dva razreda in uporabljajo CrossEntropyLoss z izbirnim label_smoothing; učne skripte izpišejo skupno in učno število parametrov.

LOSO pusti enega udeleženca za test v vsakem foldu, GROUPKFOLD pa udeležence razdeli v n_splits skupin. V vsakem foldu se validacijski udeleženci izberejo iz učne množice. Značilke se standardizirajo po kanalih in značilkah, surovi EEG po kanalih prek učnih epoh in časovnih točk; validacija in test uporabita samo učne statistike. project.random_seed nastavi semena Python, NumPy in PyTorch ter deterministične CUDA nastavitve, kadar so na voljo.

## Zagon in izhodi

Za celoten poskus:

~~~powershell
python src/train_eval/run_all_models.py
~~~

Skripta ustvari outputs/experiments/<čas>_<VALIDACIJA>_nvalsubj<N>/, nauči vse štiri modele ter po uspešnem učenju samodejno požene primerjalno analizo in statistične teste. Ob napaki se potek ustavi. experiment_timings/model_runtime_summary.csv zabeleži stanje in trajanje stopenj.

Posamezni modeli:

~~~powershell
python src/train_eval/run_cnn_eegnetlike.py
python src/train_eval/run_cnn_features1d.py
python src/train_eval/run_cnn_features2d.py
python src/train_eval/run_cnn_hybrid.py
~~~

Samostojni zagon ustvari svojo mapo. Za naknadno obdelavo uporabi:

~~~powershell
python src/train_eval/analyze_model_results.py
python src/train_eval/run_statistical_tests.py
~~~

Ročni skripti brez parametra samodejno izbereta najnovejši popoln poskus v outputs/experiments/. Za točno določen poskus podaj --experiment-dir:

~~~powershell
python src/train_eval/analyze_model_results.py --experiment-dir outputs/experiments/<čas>_<VALIDACIJA>_nvalsubj<N>
python src/train_eval/run_statistical_tests.py --experiment-dir outputs/experiments/<čas>_<VALIDACIJA>_nvalsubj<N>
~~~

~~~
outputs/experiments/<čas>_<VALIDACIJA>_nvalsubj<N>/
├── config_used.yaml
├── models/<model>/                  # .pt kontrolne točke
├── results/<model>/                 # CSV metrike, povzetki in runtime
├── plots/<model>/                   # učne krivulje, ROC, matrike zamenjav
├── analysis/                        # primerjava modelov
├── statistical_tests/               # bazne napovedi in testi
└── experiment_timings/
~~~

Metrike po foldih vključujejo testne/validacijske udeležence, najboljšo epoho, najboljše učne/validacijske vrednosti ter epoch_accuracy, epoch_precision, epoch_recall, epoch_f1 in epoch_roc_auc. Vsebujejo tudi štiri celice matrike zmot; pri save_predictions: true se shranijo še napovedi po epohah v results/<model>/predictions/. analysis vsebuje all_model_fold_results.csv, model_comparison_summary.csv, uporabljene konfiguracije in primerjalne grafe za točnost, F1 in ROC-AUC; vrstni red je Raw EEGNet-like, Features1D, Features2D, Hybrid.

statistical_tests vsebuje klasifikatorja vedno Before in vedno After ter dvostranske natančne sign-flip teste: vsak model proti naključni točnosti 0,5 in vsi pari modelov za epoch_accuracy in epoch_f1. Shranjeni sta nepopravljena p_value_two_sided in Bonferronijevo korigirana p_value_bonferroni; popravek uporablja m = 4 za štiri načrtovane primerjave modelov. LOSO teste interpretiraj po udeležencih. GroupKFold teste interpretiraj raziskovalno, ker se učne množice foldov prekrivajo.

## Gnezdena Optuna optimizacija

Pred LOSO končnim učenjem zaženi optimizacijo za vse štiri modele:

~~~powershell
python -u src/train_eval/optimize_hyperparameters.py --model raw
python -u src/train_eval/optimize_hyperparameters.py --model features1d
python -u src/train_eval/optimize_hyperparameters.py --model features2d
python -u src/train_eval/optimize_hyperparameters.py --model hybrid
~~~

--trials N prepiše število poskusov. Optimizacija uporablja zunanji LOSO in notranji GroupKFold, maksimira uravnoteženo točnost in kaznuje nestabilnost med notranjimi foldi. Rezultati so v outputs/hyperparameter_optimization/<model>/: best_params_subject_<ID>.json, trials_subject_<ID>.json in nested_optimization_summary.json.

Pomembno: trenutna implementacija optimizacije vedno filtrira skupino Caffeine. LOSO uporabi fold-specifične Optuna parametre. GROUPKFOLD uporablja eno nespremenljivo globalno konfiguracijo, agregirano iz LOSO optimumov: mediano za zvezne in modus za diskretne oziroma kategorične parametre. Artefakta global_params_from_loso.json in fold_hyperparameters.json zabeležita uporabljene nastavitve.

## EDA in čiščenje

notebooks/exploration.ipynb vsebuje sestavo zbirke, kakovost signalov, kanale, reprezentativne epohe, časovne in spektralne prikaze. Slike so v notebooks/eda_figures/.

~~~powershell
python src/utils/prune_output_folders.py --folder outputs/experiments --before 2026-08-01 --dry-run
~~~

Za brisanje sta potrebna --delete in interaktivni vnos DELETE. Surove .mat in zlasti .npz datoteke so velike, zato gradnja in učenje zahtevata precej RAM-a in časa.

## Podatkovna pogodba in zunanja predobdelava

Repozitorij ne izvaja prvotne predobdelave EEG. Vhod mora biti MATLAB datoteka z objektom ALLEEG, katerega postavke imajo naslednja polja:

| Polje | Zahteva |
| --- | --- |
| data | številčno polje (kanali, časovne_točke, epohe) |
| subject | skalarni identifikator udeleženca |
| condition | Before ali After; primerjava ne razlikuje velikosti črk |
| group | Caffeine ali Placebo |

Trenutna konfiguracija predpostavlja 600 Hz in 900 vzorcev na epoho, torej 1,5 s. Vrstni red kanalov se med gradnjo ohrani in mora biti enak za vse pogoje in udeležence. Koda ne preverja imen kanalov, montaže, referenciranja, filtrov ali odstranitve artefaktov; te odločitve je treba dokumentirati ob izvornih podatkih in uporabljati dosledno.

Končna modelska filtracija skupini Caffeine in Placebo primerja neodvisno od velikosti črk. Prazni podatkovni zapisi se preskočijo; manjkajoča ali drugače poimenovana polja povzročijo napako ali neveljavno zbirko.

## Ponovljivost in omejitve

Za popolno tehnično ponovitev obdrži ali navedi:

1. izvorno .mat datoteko oziroma njeno nedvoumno verzijo in vse korake predobdelave;
2. uporabljeni config.yaml ter različico kode;
3. zbirki v data/processed/ ali ukaz za njuno ponovno gradnjo;
4. pri LOSO vse Optuna datoteke za štiri modele;
5. celotno mapo pod outputs/experiments/.

Vsak zaključen poskus shrani config_used.yaml, metrike, grafe, analizo in trajanja. Za primerjavo rezultatov uporabi konfiguracijo v mapi poskusa, ne samo trenutne korenske konfiguracije.

Metrike so na ravni epoh. Epohe istega udeleženca niso neodvisne opazke, zato povprečnih epoch-level metrik ne interpretiraj kot število neodvisnih oseb. LOSO je primernejši za sklepanje po udeležencih. GroupKFold p-vrednosti so raziskovalne, ker se učne množice foldov prekrivajo. Optuna optimizacija za LOSO je trenutno implementirana samo za skupino Caffeine; placebo ali združena analiza potrebujeta prilagoditev oziroma ločeno preverjeno optimizacijo.

Projekt razvršča Before proti After in sam po sebi ne dokazuje vzročne spremembe zaradi kofeina. Pri interpretaciji upoštevaj predobdelavo, neravnotežje epoh, majhno število udeležencev in individualne razlike.

## Strojne zahteve, zasebnost in arhiviranje

CPU zadošča; training.device: auto izbere CUDA, kadar jo PyTorch zazna. Učenje surovega EEG in zlasti gnezdena optimizacija sta računsko zahtevna. Stisnjena surova zbirka je velika približno 391 MB, ob nalaganju in standardizaciji pa potrebuje več RAM-a kot na disku. Najprej preveri potek z enim modelom ali manj Optuna poskusi, dejanske čase pa preglej v experiment_timings/model_runtime_summary.csv.

Surovi podatki niso v repozitoriju. Pred deljenjem .mat, .npz, kontrolnih točk ali izhodov preveri dovoljenja, soglasja, identifikatorje udeležencev in politiko hrambe ustanove. Za dolgoročni arhiv obdrži celotno mapo eksperimenta, pripadajoče Optuna rezultate in dokumentacijo predobdelave.
