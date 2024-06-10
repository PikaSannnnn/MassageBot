import time
import torch
import tqdm

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

def train(model, optimizer, criterion, train_loader, val_loader, anchors, epochs=10):
    print('Criterion Grad:', criterion.requires_grad)
    
    model = model.to(device)
    val_loss = []
    train_losses = []
    for epoch in tqdm.tqdm(range(epochs), desc='Yolov3 Training'):
        train_loss = 0
        start_time = time.time()
        for i, (imgs, labels) in enumerate(train_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            model.train()
            
            outputs = model(imgs)
            loss = criterion(anchors, labels, outputs)            
            
            optimizer.zero_grad()
            # print(loss.item())
            train_loss += loss.item()
        
            loss.backward()
            optimizer.step()
                
            # Delete stuff to free up resources
            torch.cuda.empty_cache()
            # del imgs
            # del labels
            # del outputs
            # del loss
            
        avg_train_loss = train_loss/len(train_loader)
        train_losses.append(avg_train_loss)
        end_time = time.time()
        avg_val_loss = validate(model, criterion, val_loader, anchors, device)
        val_loss.append(avg_val_loss)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_train_loss:.4f}, Val_loss: {avg_val_loss:.4f}, Time: {end_time - start_time:.2f}")
    return train_losses, val_loss

def validate(model, criterion, val_loader, anchors, device):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for i, (imgs, labels) in enumerate(val_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(anchors, labels, outputs)
            val_loss += loss.item()
    return val_loss/(i+1)