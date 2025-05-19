import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import random_split

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
    def __init__(self, data):

        
        features = [eval(x) for x in data["Decoder"].values]
        labels = [eval(x) for x in data["Encoder"].values]
        
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        

    

        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
    
from utils import load_data
    



    
# GRU tabanlı model
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(hidden_size, int(hidden_size/2))
        self.fc2 = nn.Linear(int(hidden_size/2), hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Boyut düzenleme: [batch, features] -> [batch, seq_len=1, features]
        if len(x.shape) == 2:
            x = x.unsqueeze(1)

        # print(x.shape)
            
        # GRU çıktısı: [batch, seq_len, hidden_size]
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.gru(x, h0)
        out = self.relu(out)
        # Son zaman adımının çıktısını al
        out = self.fc(out[:, -1, :])
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.sigmoid(out)
        return out
    

model = GRUModel(647, 1024, 2, 647)




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
        features = [eval(x) for x in data.iloc[:, :-1].values]
        labels = data.iloc[:, -1].values
    
    # Verileri ölçeklendirme
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42
    )
    
    return X_train, X_test, y_train, y_test, scaler

dataset = LDPCDataset(load_data("/home/ayoldass/Masaüstü/aff3ct_gru/data1.csv"))



train_set,val_set,test_set = random_split(dataset, [80, 24,20])

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader = DataLoader(val_set, batch_size=32, shuffle=False)

x = next(iter(train_loader))
print(x[0].shape)



def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=10000):
    train_losses = []
    val_losses = []
    acc_up = 0
    acc_down = 0
    
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
            acc_out = outputs.squeeze().detach().numpy()
            acc_out = np.where(acc_out > 0.6, 1., 0.)
            labels = labels.detach().cpu().numpy()

            acc_up += np.sum(acc_out == labels)
            acc_down += (labels.shape[0]*labels.shape[1])
        
        
        
            
            running_loss += loss.item()
        acc_train = acc_up / acc_down

        acc_up = 0
        acc_down = 0
        
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
                acc_out = outputs.squeeze().cpu().numpy()
                acc_out = np.where(acc_out > 0.5, 1., 0.)
                labels = labels.cpu().numpy()

                acc_up += np.sum(acc_out == labels)
                acc_down += (labels.shape[0]*labels.shape[1])

            
                val_loss += loss.item()
        acc = acc_up / acc_down
        
        acc_up = 0
        acc_down = 0  
        epoch_val_loss = val_loss / len(val_loader)
        val_losses.append(epoch_val_loss)
        
        
        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f},Train acc : {acc_train:.4f}, Val acc : {acc:.4f}' )
    
    return train_losses, val_losses

train_loss,val_loss = train_model(model=model, train_loader=train_loader, val_loader=val_loader, criterion=nn.BCELoss(), optimizer=optim.Adam(model.parameters(), lr=0.001), device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), num_epochs=1000)
#     return X_train, X_test, y_train, y_test

# Model değerlendirme fonksiyonu
# Burası düzenlenecek bitlerin doğruluğu kontrol edilecek?
# def evaluate_model(model, test_loader, device):
#     model.eval()
#     correct = 0
#     total = 0
#     predictions = []
#     true_labels = []
    
#     with torch.no_grad():
#         for inputs, labels in test_loader:
#             inputs, labels = inputs.to(device), labels.to(device)
#             outputs = model(inputs)
#             predicted = (outputs.squeeze() > 0.5).float()
            
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
            
#             predictions.extend(predicted.cpu().numpy())
#             true_labels.extend(labels.cpu().numpy())
    
#     accuracy = 100 * correct / total
#     print(f'Test Accuracy: {accuracy:.2f}%')
    
#     return predictions, true_labels, accuracy

# # Ana fonksiyon
# def main(file_path="YN_output.xlsx"):
#     # Cihaz seçimi
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Kullanılan cihaz: {device}")
    
#     # Veriyi yükle
#     data = load_data(file_path)
#     if data is None:
#         return
    
    
#     # Dataset ve DataLoader oluşturma
#     train_dataset = LDPCDataset(X_train, y_train)
#     test_dataset = LDPCDataset(X_test, y_test)
    
#     train_size = int(0.8 * len(train_dataset))
#     val_size = len(train_dataset) - train_size
#     train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    
#     batch_size = 64
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     val_loader = DataLoader(val_dataset, batch_size=batch_size)
#     test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
#     # Model parametreleri
#     input_size = X_train.shape[1]  # Özellik sayısı
#     hidden_size = 128
#     num_layers = 2
#     output_size = 1


#     class Model(nn.Module):
#         def __init__(self, input_size, hidden_size, num_layers, output_size):
#             super(Model, self).__init__()

#             self.embedding = nn.Embedding(input_size, hidden_size)

#             self.gru = nn.GRU(hidden_size, int(hidden_size/2), num_layers, batch_first=True)
#             self.fc = nn.Linear(hidden_size, output_size)
#             self.sigmoid = nn.Sigmoid()

#         def forward(self, x):
#             h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
#             out = self.embedding(x)
#             out = out.view(x.size(0), 1, -1)
#             out, _ = self.gru(out, h0)
#             out = self.fc(out[:, -1, :])
#             out = self.sigmoid(out)
#             return out
    
#     # Model oluşturma
#     # model = GRUModel(input_size, hidden_size, num_layers, output_size).to(device)
#     # print(model)

#     model = Model(input_size, hidden_size, num_layers, output_size).to(device)
#     print(model)
    
#     # Kayıp fonksiyonu ve optimizer
#     criterion = nn.MSELoss()
#     optimizer = optim.Adam(model.parameters(), lr=0.001)
    
#     # Model eğitimi
#     train_losses, val_losses = train_model(
#         model, train_loader, val_loader, criterion, optimizer, device, num_epochs=100
#     )
    
#     # Model değerlendirme
#     predictions, true_labels, accuracy = evaluate_model(model, test_loader, device)
    
#     # Eğitim sürecini görselleştirme
#     plt.figure(figsize=(10, 5))
#     plt.plot(train_losses, label='Training Loss')
#     plt.plot(val_losses, label='Validation Loss')
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
#     plt.title('Training and Validation Loss')
#     plt.legend()
#     plt.savefig('training_loss.png')
#     plt.show()
    
#     # Modeli kaydetme
#     torch.save(model.state_dict(), 'ldpc_gru_model.pth')
#     print("Model başarıyla kaydedildi: ldpc_gru_model.pth")

# if __name__ == "__main__":
#     main()