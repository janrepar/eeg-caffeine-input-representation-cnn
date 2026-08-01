# Spremembe projekta

## 1.0.2 - 30. julij 2026

- Izpisi učenja, modeli, rezultati, grafi, primerjalna analiza, časovne meritve in LOSO statistični testi se zdaj shranjujejo v enotno, datumsko urejeno mapo posameznega eksperimenta (`outputs/experiments/<datum>_<validacija>_nvalsubj<N>/`).
- `run_all_models.py` ustvari eno skupno mapo eksperimenta za vse štiri arhitekture in po uspešnem učenju samodejno zažene primerjalno analizo ter statistične teste.
- Statistični testi podpirajo LOSO in GroupKFold; pri GroupKFold se primerjave izvajajo po ujemajočih se foldih in so označene kot raziskovalne zaradi prekrivajočih se učnih množic.
- Primerjalni grafi modelov uporabljajo stalni vrstni red: Raw EEGNet-like, Features1D, Features2D, Hybrid.
- Učne skripte izpišejo skupno in število parametrov ustvarjenega modela.
- EDA notebook je razširjen z opisi sestave podatkov, kakovosti signalov, časovnih in spektralnih prikazov ter belim ozadjem vseh figur.

## 1.0.1 - 18. julij 2026

- Popravljen prehod modela iz evalvacijskega v učni režim po vmesni validaciji.
- Dodano enotno nastavljanje psevdonaključnih semen za Python, NumPy in PyTorch.
- Iz frekvenčnih značilk odstranjen delta pas, ker 1,5-sekundne epohe ne omogočajo zanesljive ocene pasu do 0,5 Hz.
- Parameter `welch_nperseg` se zdaj dejansko uporablja pri oceni spektralne moči in spektralne entropije; nastavljen je na celotno epoho (900 vzorcev).
- Moč frekvenčnih pasov se zdaj pretvori z `log10(power + 1e-12)`.
- Odstranjeni sta redundantni časovni značilki: varianca in Hjorthova aktivnost; standardni odklon ostane kot amplitudna značilka.
- Funkciji `train_multiclass_model` in `train_hybrid_multiclass_model` sta preimenovani v `train_model` in `train_hybrid_model`.
- Primerjava modelov zdaj izbere najnovejšo mapo, ki vsebuje CSV za zahtevano validacijsko shemo, zato modeli niso tiho izpuščeni ob sočasnih LOSO in GroupKFold rezultatih.
- Dodana skripta `run_statistical_tests.py` za referenčna klasifikatorja in natančne permutacijske teste na LOSO rezultatih po udeležencih.
- Popravljeno usklajevanje imen in vrednosti časovnih značilk po odstranitvi variance.
- Primarna validacijska shema je nastavljena na LOSO za udeleženčevo neodvisno evalvacijo in statistične teste.
- Dodana odvisnost Optuna (4.x) za gnezdeno optimizacijo hiperparametrov.
- Dodana `optimize_hyperparameters.py`: gnezdena Optuna optimizacija (zunanji LOSO, notranji GroupKFold) za vse štiri arhitekture.
- Končne učne skripte zdaj pri vsakem LOSO foldu zahtevajo in uporabijo Optuna parametre za ustreznega testnega udeleženca.

## 1.0.0 - začetna različica

- Začetna različica projekta za primerjavo CNN-modelov na surovem EEG, značilkah in kombiniranem vhodu.