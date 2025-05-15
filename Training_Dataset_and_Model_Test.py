import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

# Ana model scriptinden sınıfları içe aktarma
from LDPC_GRU_Training_Model import GRUModel, LDPCDataset

def prepare_data_from_excel(file_path):
    """
    Excel dosyasını okuyarak eğitim için veri hazırlar
    """
    print(f"'{file_path}' dosyası okunuyor...")
    
    try:
        # Excel dosyasını oku
        data = pd.read_excel(file_path)
        print(f"Verinin boyutu: {data.shape}")
        print("Sütun isimleri:", data.columns.tolist())
        
        # İlk birkaç satırı göster
        print("\nVeri önizlemesi:")
        print(data.head())
        
        # Veri tipi bilgisi
        print("\nVeri tipleri:")
        print(data.dtypes)
        
        # Eksik değerleri kontrol et
        print("\nEksik değerler:")
        print(data.isnull().sum())
        
        # İstatistiksel özet
        print("\nİstatistiksel özet:")
        print(data.describe())
        
        return data
    
    except Exception as e:
        print(f"Hata: {e}")
        return None

def analyze_data_distribution(data):
    """
    Veri dağılımını analiz eder ve görselleştirir
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Histogram for each feature
    plt.figure(figsize=(15, 10))
    for i, column in enumerate(data.columns):
        if i >= 16:  # Limit the number of plots
            break
        plt.subplot(4, 4, i+1)
        sns.histplot(data[column], kde=True)
        plt.title(f'{column} Distribution')
        plt.tight_layout()
    
    plt.savefig('feature_distributions.png')
    plt.close()
    
    # Correlation matrix
    plt.figure(figsize=(12, 10))
    corr = data.corr()
    sns.heatmap(corr, annot=False, cmap='coolwarm', fmt='.2f')
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig('correlation_matrix.png')
    plt.close()
    
    print("Veri dağılım grafikleri oluşturuldu.")

def test_trained_model(model_path, test_loader, device):
    """
    Eğitilmiş modeli test eder ve sonuçları değerlendirir
    """
    # Modeli yükle
    input_size = next(iter(test_loader))[0].shape[1]
    model = GRUModel(input_size=input_size, hidden_size=128, num_layers=2, output_size=1)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    
    # Test değerlendirmesi
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            predicted = (outputs.squeeze() > 0.5).float()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')
    
    # Confusion Matrix
    from sklearn.metrics import confusion_matrix, classification_report
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    # Classification Report
    report = classification_report(all_labels, all_preds)
    print("\nClassification Report:")
    print(report)
    
    return accuracy, all_preds, all_labels

def prepare_custom_input(scaler, custom_data):
    """
    Özel bir giriş verisini model için hazırlar
    """
    # Özellik vektörünü standartlaştır
    scaled_data = scaler.transform(custom_data)
    
    # Tensor'a dönüştür
    tensor_data = torch.tensor(scaled_data, dtype=torch.float32)
    
    return tensor_data

def predict_single_sample(model, input_tensor, device):
    """
    Tek bir örnek için tahmin yapar
    """
    model.eval()
    input_tensor = input_tensor.to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
        prediction = (output.squeeze() > 0.5).float()
    
    return prediction.item(), output.item()

if __name__ == "__main__":
    # Excel dosyasını oku
    data = prepare_data_from_excel("YN_output.xlsx")
    
    if data is not None:
        # Veri dağılımını analiz et
        analyze_data_distribution(data)
        
        print("\nVeri analizi tamamlandı. Şimdi modeli test etmek için ana scripti çalıştırabilirsiniz.")
        print("python ldpc_gru_model.py")