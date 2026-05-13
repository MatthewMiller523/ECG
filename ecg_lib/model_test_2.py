# -*- coding: utf-8 -*-
#=============================
# test

import torch
import torch.nn as nn


def m_test_1(test_loader, t_model):
    #================================================================
    '''
    print(f"len(test_loader.dataset): {len(test_loader.dataset)}")
    print(f"len(test_loader): {len(test_loader)}")

    if len(test_loader.dataset) == 0:
        raise ValueError("test_loader.dataset is empty")

    if len(test_loader) == 0:
        raise ValueError("test_loader has zero batches")
    '''
    #========================================================================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    t_model = t_model.to(device)
    criterion = nn.BCEWithLogitsLoss()

    t_model.eval()

    test_loss = 0.0

    correct_labels = 0
    total_labels = 0

    correct_samples = 0
    total_samples = 0

    with torch.no_grad():
        for X_batch, Y_batch, Z_batch in test_loader:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device).float()
            Z_batch = Z_batch.to(device)

            outputs = t_model(X_batch, Z_batch)
            loss = criterion(outputs, Y_batch)

            test_loss += loss.item()

            probabilities = torch.sigmoid(outputs)
            predicted = (probabilities >= 0.5).float()

            correct_labels += (predicted == Y_batch).sum().item()
            total_labels += Y_batch.numel()

            correct_samples += (predicted == Y_batch).all(dim=1).sum().item()
            total_samples += Y_batch.size(0)
    #========================================================
    '''
    print(f"total_labels: {total_labels}")
    print(f"total_samples: {total_samples}")
    
    if total_labels == 0:
        raise ValueError("No test labels were evaluated")

    if total_samples == 0:
        raise ValueError("No test samples were evaluated")
        '''
    #=======================================================
    avg_test_loss = test_loss / len(test_loader)
    label_accuracy = correct_labels / total_labels
    exact_match_accuracy = correct_samples / total_samples

    print(f"Test loss: {avg_test_loss:.4f}")
    print(f"Per-label test accuracy: {label_accuracy:.4f}")
    print(f"Exact-match test accuracy: {exact_match_accuracy:.4f}")

    return avg_test_loss, label_accuracy, exact_match_accuracy