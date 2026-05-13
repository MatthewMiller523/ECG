# -*- coding: utf-8 -*-
#=============================
#test
import torch
import torch.nn as nn

def m_test_1(test_loader, t_model):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    t_model = t_model.to(device)
    criterion = nn.CrossEntropyLoss()
    
    t_model.eval()

    test_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for X_batch, Y_batch in test_loader:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.long().to(device)
            Z_batch = Z_batch.to(device)
            
            outputs = t_model(X_batch, Z_batch)
            loss = criterion(outputs, Y_batch)
            
            test_loss += loss.item()
            
            predicted = torch.argmax(outputs, dim=1)
            correct += (predicted == Y_batch).sum().item()
            total += Y_batch.size(0)
            
    avg_test_loss = test_loss / len(test_loader)
    test_accuracy = correct / total
    
    print(f"Test loss: {avg_test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    
    return avg_test_loss, test_accuracy
