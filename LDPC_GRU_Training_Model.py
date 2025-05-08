import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#print(torch.cuda.is_available())

# Excel dosyasını okuma
def load_data(file_path):
    try:
        data = pd.read_excel(file_path)
        print(f"Veri başarıyla yüklendi. Veri şekli: {data.shape}")
        print("İlk 5 satır:")
        print(data.head())
        return data
    except Exception as e:
        print(f"Veri yükleme hatası: {e}")
        return None

# Özel Dataset sınıfı
class LDPCDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# GRU tabanlı model
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Boyut düzenleme: [batch, features] -> [batch, seq_len=1, features]
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
            
        # GRU çıktısı: [batch, seq_len, hidden_size]
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)
        
        # Son zaman adımının çıktısını al
        out = self.fc(out[:, -1, :])
        out = self.sigmoid(out)
        return out

# Veri önişleme fonksiyonu
def preprocess_data(data, sequence_length=1):
    # Burada veri yapınıza özel işlemler yapmanız gerekebilir
    # Excel dosyasının içeriğine göre düzenleyin
    
    # Örnek: İlk sütun hariç tüm sütunları özellik olarak kullan
    if 'label' in data.columns:
        # Label sütunu varsa
        features = data.drop('label', axis=1).values
        labels = data['label'].values
    else:
        # Veri yapısına göre ayarlayın - örnek olarak son sütunu etiket kabul edelim
        features = data.iloc[:, :-1].values
        labels = data.iloc[:, -1].values
    
    # Verileri ölçeklendirme
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42
    )
    
    return X_train, X_test, y_train, y_test, scaler

# Model eğitim fonksiyonu
def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=50):
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        epoch_train_loss = running_loss / len(train_loader)
        train_losses.append(epoch_train_loss)
        
        # Doğrulama
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), labels)
                val_loss += loss.item()
        
        epoch_val_loss = val_loss / len(val_loader)
        val_losses.append(epoch_val_loss)
        
        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}')
    
    return train_losses, val_losses

# Model değerlendirme fonksiyonu
# Burası düzenlenecek bitlerin doğruluğu kontrol edilecek?
def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            predicted = (outputs.squeeze() > 0.5).float()
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            predictions.extend(predicted.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')
    
    return predictions, true_labels, accuracy

# Ana fonksiyon
def main(file_path="YN_output.xlsx"):
    # Cihaz seçimi
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Kullanılan cihaz: {device}")
    
    # Veriyi yükle
    data = load_data(file_path)
    if data is None:
        return
    
    # Veri önişleme
    X_train, X_test, y_train, y_test, scaler = preprocess_data(data)
    
    # Dataset ve DataLoader oluşturma
    train_dataset = LDPCDataset(X_train, y_train)
    test_dataset = LDPCDataset(X_test, y_test)
    
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Model parametreleri
    input_size = X_train.shape[1]  # Özellik sayısı
    hidden_size = 128
    num_layers = 2
    output_size = 1
    
    # Model oluşturma
    model = GRUModel(input_size, hidden_size, num_layers, output_size).to(device)
    print(model)
    
    # Kayıp fonksiyonu ve optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Model eğitimi
    train_losses, val_losses = train_model(
        model, train_loader, val_loader, criterion, optimizer, device, num_epochs=100
    )
    
    # Model değerlendirme
    predictions, true_labels, accuracy = evaluate_model(model, test_loader, device)
    
    # Eğitim sürecini görselleştirme
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('training_loss.png')
    plt.show()
    
    # Modeli kaydetme
    torch.save(model.state_dict(), 'ldpc_gru_model.pth')
    print("Model başarıyla kaydedildi: ldpc_gru_model.pth")

if __name__ == "__main__":
    main()