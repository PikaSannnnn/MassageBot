import time
import torch

def train(model, optimizer, criterion, train_loader, val_loader, epochs, device):
    val_loss = []
    train_losses = []
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        start_time = time.time()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        avg_train_loss = train_loss/len(train_loader)
        train_losses.append(avg_train_loss)
        end_time = time.time()
        avg_val_loss = validate(model, criterion, val_loader, device)
        val_loss.append(avg_val_loss)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_train_loss:.4f}, Val_loss: {avg_val_loss:.4f}, Time: {end_time - start_time:.2f}")
    return train_losses, val_loss

def validate(model, criterion, val_loader, device):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for i, (imgs, labels) in enumerate(val_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    return val_loss/(i+1)