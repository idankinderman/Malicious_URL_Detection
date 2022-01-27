# -*- coding: utf-8 -*-
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import plotly
# optuna
import optuna
from preprocessing import BuildDataLoaderNoTokenizatie
from training_and_evaluating import Train, DrawGraphs
from embedding_and_positional_encoding import Embeddings
from transformer import TransformerClassifier



# some definitions
batch_size = 32
classes = 2
epochs = 5
log_interval = 10
n_train_examples = batch_size * 120
n_valid_examples = batch_size * 80


def objective(trial):
    
    # samples the hyper parameters
    num_layers = trial.suggest_int("num_layers", 1, 3)  # number of layers will be between 1 and 3
    num_heads = trial.suggest_int("num_heads", 6, 12, step=2)
    d_model = trial.suggest_int("d_model", 14, 32, step=2)
    conv_hidden_dim = trial.suggest_int("conv_hidden_dim", 90, 180)
    dropout = trial.suggest_float("dropout", 0.05, 0.4)
    lr = trial.suggest_float("lr", 0.0001, 0.05, log=True)  # log=True, will use log scale to interplolate between lr
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD"])
    
    if d_model % num_heads != 0:
        raise optuna.exceptions.TrialPruned()
    # Generate the model.
    model = TransformerClassifier(num_layers, d_model, num_heads, conv_hidden_dim, num_answers=2, dropout=dropout)
    # Generate the optimizers.
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

    # Get the dataset.
    train_loader, valid_loader, _ = BuildDataLoaderNoTokenizatie(url_num=30_000, batch_size=32, num_classes=2)

    # Training of the model.
    for epoch in range(epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            # Limiting training data for faster epochs.
            if batch_idx * batch_size >= n_train_examples:
                break

            #data, target = data.view(data.size(0), -1), target
            optimizer.zero_grad()
            output = model(data)
            loss = f.cross_entropy(output, target)
            loss.backward()
            optimizer.step()

        # Validation of the model.
        model.eval()
        correct = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(valid_loader):
                # Limiting validation data.
                if batch_idx * batch_size >= n_valid_examples:
                    break
                output = model(data)
                # Get the index of the max log-probability.
                pred = output.argmax(dim=1, keepdim=True)
                correct += (pred == target).cpu().numpy().sum().item()

        accuracy = correct / min(len(valid_loader.dataset), n_valid_examples)
        print("accuracy", accuracy)

        # report back to Optuna how far it is (epoch-wise) into the trial and how well it is doing (accuracy)
        trial.report(accuracy, epoch)  

        # then, Optuna can decide if the trial should be pruned
        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return accuracy




# now we can run the experiment
sampler = optuna.samplers.TPESampler()
study = optuna.create_study(study_name="transformer URL", direction="maximize", sampler=sampler)
study.optimize(objective, n_trials=200)

pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

print("Study statistics: ")
print("  Number of finished trials: ", len(study.trials))
print("  Number of pruned trials: ", len(pruned_trials))
print("  Number of complete trials: ", len(complete_trials))

print("\n\n\nBest trial:")
trial = study.best_trial

print("  Value: ", trial.value)

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))
    
optuna.visualization.plot_param_importances(study)

optuna.visualization.plot_contour(study, params=["num_layers", "num_heads"])
optuna.visualization.plot_contour(study, params=["d_model", "conv_hidden_dim"])
optuna.visualization.plot_contour(study, params=["lr", "dropout"])
optuna.visualization.plot_contour(study, params=["num_layers", "d_model"])
optuna.visualization.plot_contour(study, params=["num_heads", "conv_hidden_dim"])