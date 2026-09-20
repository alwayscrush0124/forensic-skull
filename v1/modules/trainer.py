import torch, torch.nn as nn, torch.optim as optim
import matplotlib.pyplot as plt

def train_with_checkpoint(model, loader, epochs=5, save_path='checkpoints/best_model.pth'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    best_loss = float('inf')
    history = []
    for epoch in range(epochs):
        model.train(); running_loss = 0.0
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            out, _ = model(inputs)
            loss = criterion(out, targets)
            loss.backward(); optimizer.step(); running_loss += loss.item()
        avg_loss = running_loss / len(loader); history.append(avg_loss)
        print(f"Epoch {epoch+1}/5 | Loss: {avg_loss:.4f}")
        if avg_loss < best_loss:
            best_loss = avg_loss; torch.save(model.state_dict(), save_path)
            print(f"⭐ 模型已儲存至 {save_path}")
    return history
