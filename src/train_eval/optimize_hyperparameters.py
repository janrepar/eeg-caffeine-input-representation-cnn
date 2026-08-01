"""Nested Optuna hyperparameter optimisation for participant-independent EEG models.
Run one model at a time, e.g.:
python -u src/train_eval/optimize_hyperparameters.py --model features2d --trials 60
"""
import argparse, json, os, sys
from pathlib import Path

sys.path.append(os.path.abspath('.'))
import numpy as np
import optuna
import torch
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold

from src.models.cnn_eegnetlike import EEGNetLike
from src.models.cnn_features_1d import CNNFeatures1D
from src.models.cnn_features_2d import CNNFeatures2D
from src.models.cnn_hybrid import CNNHybrid
from src.train_eval.common import (load_dataset_npz, get_device, set_random_seed,
    standardize_raw_train_val_test, standardize_features_train_val_test)
from src.train_eval.train import train_model, train_hybrid_model
from src.train_eval.evaluate import predict_multiclass_model, predict_hybrid_multiclass_model
from src.utils.helpers import load_config


def suggest(trial, model):
    """Suggest a deliberately regularised search space for small EEG datasets."""
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-5, 3e-3, log=True),
        "dropout": trial.suggest_float("dropout", 0.35, 0.70),
        "label_smoothing": trial.suggest_float("label_smoothing", 0.0, 0.10),
    }

    if model == "raw":
        params |= {"temporal_filters": trial.suggest_categorical("temporal_filters", [8, 16]),
                   "depth_multiplier": trial.suggest_categorical("depth_multiplier", [1, 2]),
                   "temporal_kernel_size": trial.suggest_categorical("temporal_kernel_size", [64, 128, 300]),
                   "separable_kernel_size": trial.suggest_categorical("separable_kernel_size", [8, 16])}
    elif model in {"features1d", "features2d"}:
        params |= {"conv1_filters": trial.suggest_categorical("conv1_filters", [8, 16]),
                   "conv2_filters": trial.suggest_categorical("conv2_filters", [16, 32]),
                   "hidden_units": trial.suggest_categorical("hidden_units", [16, 32])}
    else:
        params |= {"raw_temporal_filters": trial.suggest_categorical("raw_temporal_filters", [8,16]),
                   "feature_conv1_filters": trial.suggest_categorical("feature_conv1_filters", [8,16]),
                   "feature_conv2_filters": trial.suggest_categorical("feature_conv2_filters", [16,32]),
                   "raw_embedding_units": trial.suggest_categorical("raw_embedding_units", [16,32]),
                   "feature_embedding_units": trial.suggest_categorical("feature_embedding_units", [16,32]),
                   "fusion_hidden_units": trial.suggest_categorical("fusion_hidden_units", [16,32])}
    return params


def make_model(kind, raw_shape, feature_shape, p):
    channels, timepoints = raw_shape[1:]; features = feature_shape[2]

    if kind == "raw": return EEGNetLike(channels,timepoints,2, p["dropout"],p["temporal_filters"],p["depth_multiplier"],p["temporal_kernel_size"],p["separable_kernel_size"])
    if kind == "features1d": return CNNFeatures1D(channels,features,2,p["conv1_filters"],p["conv2_filters"],3,3,p["hidden_units"],p["dropout"])
    if kind == "features2d": return CNNFeatures2D(channels,features,2,p["conv1_filters"],p["conv2_filters"],(3,3),(3,3),(2,1),(2,1),p["hidden_units"],p["dropout"])

    return CNNHybrid(channels,timepoints,features,2,p["raw_temporal_filters"],300,16,p["feature_conv1_filters"],p["feature_conv2_filters"],3,3,p["raw_embedding_units"],p["feature_embedding_units"],p["fusion_hidden_units"],p["dropout"])


def score_fold(kind, raw, feat, y, train_idx, valid_idx, p, cfg, device, seed):
    set_random_seed(seed)
    rtr,rva,_=standardize_raw_train_val_test(raw[train_idx],raw[valid_idx],raw[valid_idx])
    ftr,fva,_=standardize_features_train_val_test(feat[train_idx],feat[valid_idx],feat[valid_idx])
    xrtr=torch.tensor(rtr).unsqueeze(1); xrva=torch.tensor(rva).unsqueeze(1)
    xftr=torch.tensor(ftr).unsqueeze(1) if kind=="features2d" else torch.tensor(ftr)
    xfva=torch.tensor(fva).unsqueeze(1) if kind=="features2d" else torch.tensor(fva)
    yt=torch.tensor(y[train_idx],dtype=torch.long); yv=torch.tensor(y[valid_idx],dtype=torch.long)
    model=make_model(kind,raw.shape,feat.shape,p)
    kwargs=dict(device=device,epochs=cfg["epochs"],batch_size=cfg["batch_size"],learning_rate=p["learning_rate"],weight_decay=p["weight_decay"],patience=cfg["patience"],validation_frequency=cfg["validation_frequency"],early_stopping_metric=cfg["early_stopping_metric"],label_smoothing=p["label_smoothing"])
    
    if kind=="hybrid":
        model,_=train_hybrid_model(model,xrtr,xftr,yt,xrva,xfva,yv,**kwargs); prob,pred=predict_hybrid_multiclass_model(model,xrva,xfva,device)
    else:
        xtr=xrtr if kind=="raw" else xftr; xva=xrva if kind=="raw" else xfva
        model,_=train_model(model,xtr,yt,xva,yv,**kwargs); prob,pred=predict_multiclass_model(model,xva,device)
    return balanced_accuracy_score(y[valid_idx],pred)

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--model",choices=["raw","features1d","features2d","hybrid"],required=True); parser.add_argument("--trials",type=int,default=None); args=parser.parse_args()
    config=load_config("config.yaml"); set_random_seed(config["project"]["random_seed"]); device=get_device(config)
    raw,y,subjects,_,groups=load_dataset_npz(config["data"]["raw_dataset_output"]); feat,y2,s2,_,g2=load_dataset_npz(config["data"]["feature_dataset_output"])
    mask=np.char.lower(groups.astype(str))=="caffeine"; raw,y,subjects,feat=raw[mask],y[mask],subjects[mask],feat[mask]

    if not (np.array_equal(y,y2[mask]) and np.array_equal(subjects,s2[mask])): raise ValueError("Raw and feature datasets are not aligned.")
    hpo=config.get("hyperparameter_optimization",{}); inner=GroupKFold(n_splits=hpo.get("inner_splits",3)); train_cfg={"epochs":hpo.get("epochs",100),"batch_size":config["training"]["batch_size"],"patience":hpo.get("patience",30),"validation_frequency":config["training"]["validation_frequency"],"early_stopping_metric":config["training"].get("early_stopping_metric","val_loss")}; n_trials=args.trials if args.trials is not None else hpo.get("trials",60)
    out=Path(config["outputs"]["output_dir"])/"hyperparameter_optimization"/args.model; out.mkdir(parents=True,exist_ok=True)
    results=[]

    for outer_subject in np.unique(subjects):
        outer_train=np.flatnonzero(subjects!=outer_subject); local_groups=subjects[outer_train]
        def objective(trial):
            p=suggest(trial,args.model); scores=[]
            for i,(a,b) in enumerate(inner.split(outer_train,groups=local_groups)):
                scores.append(score_fold(args.model,raw,feat,y,outer_train[a],outer_train[b],p,train_cfg,device,config["project"]["random_seed"]+i))
            mean_score=float(np.mean(scores)); std_score=float(np.std(scores))
            trial.set_user_attr("mean_inner_balanced_accuracy", mean_score)
            trial.set_user_attr("std_inner_balanced_accuracy", std_score)
            return mean_score - hpo.get("stability_penalty", 0.10) * std_score

        sampler=optuna.samplers.TPESampler(seed=config["project"]["random_seed"], multivariate=True, n_startup_trials=5)
        study=optuna.create_study(direction="maximize",sampler=sampler)
        study.optimize(objective,n_trials=n_trials)
        best_trial=study.best_trial
        record={"outer_test_subject":str(outer_subject),"best_inner_balanced_accuracy":best_trial.user_attrs["mean_inner_balanced_accuracy"],"inner_balanced_accuracy_std":best_trial.user_attrs["std_inner_balanced_accuracy"],"selection_score":best_trial.value,"best_params":best_trial.params}; results.append(record)
        (out/f"best_params_subject_{outer_subject}.json").write_text(json.dumps(record,indent=2),encoding="utf-8")
        trial_records=[{"number":trial.number,"state":trial.state.name,"selection_score":trial.value,"mean_inner_balanced_accuracy":trial.user_attrs.get("mean_inner_balanced_accuracy"),"std_inner_balanced_accuracy":trial.user_attrs.get("std_inner_balanced_accuracy"),"params":trial.params} for trial in study.trials]
        (out/f"trials_subject_{outer_subject}.json").write_text(json.dumps(trial_records,indent=2),encoding="utf-8")

    (out/"nested_optimization_summary.json").write_text(json.dumps(results,indent=2),encoding="utf-8")

if __name__=="__main__": main()
