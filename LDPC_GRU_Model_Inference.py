import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Ana model scriptinden sınıfları içe aktarma
from LDPC_GRU_Training_Model import GRUModel

class LDPCDecoder:
    def __init__(self, model_path, input_size):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model(model_path, input_size)
        self.scaler = None
    
    def load_model(self, model_path, input_size):
        """Eğitilmiş modeli yükler"""
        model = GRUModel(input_size=input_size, hidden_size=128, num_layers=2, output_size=1)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        print(f"Model yüklendi: {model_path}")
        return model
    
    def fit_scaler(self, data):
        """Verileri ölçeklendirmek için scaler'ı eğitir"""
        self.scaler = StandardScaler()
        self.scaler.fit(data)
        print("Scaler eğitildi")
    
    def decode(self, ldpc_input):
        """LDPC kodunu çözmek için GRU modelini kullanır"""
        if self.scaler is None:
            raise ValueError("Önce fit_scaler metodu ile scaler'ı eğitmelisiniz!")
        
        # Girişi ölçeklendir
        scaled_input = self.scaler.transform(ldpc_input)
        input_tensor = torch.tensor(scaled_input, dtype=torch.float32).to(self.device)
        
        # Model tahmini
        with torch.no_grad():
            output = self.model(input_tensor)
            predictions = (output.squeeze() > 0.5).float()
        
        return predictions.cpu().numpy(), output.squeeze().cpu().numpy()
    
    def batch_decode(self, test_data, true_labels=None):
        """Bir test veri seti üzerinde toplu tahmin yapar"""
        # Tahminleri al
        predictions, prob_outputs = self.decode(test_data)
        
        results = {
            'predictions': predictions,
            'probabilities': prob_outputs
        }
        
        # Eğer gerçek etiketler verildiyse performansı değerlendir
        if true_labels is not None:
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
            
            accuracy = accuracy_score(true_labels, predictions)
            precision = precision_score(true_labels, predictions, zero_division=0)
            recall = recall_score(true_labels, predictions, zero_division=0)
            f1 = f1_score(true_labels, predictions, zero_division=0)
            
            try:
                auc = roc_auc_score(true_labels, prob_outputs)
            except:
                auc = 0.0
                
            metrics = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'auc': auc
            }
            
            results['metrics'] = metrics
            print(f"Model Performansı:")
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  F1 Score: {f1:.4f}")
            print(f"  AUC: {auc:.4f}")
            
            # Confusion Matrix
            self.plot_confusion_matrix(true_labels, predictions)
            
            # ROC Curve
            self.plot_roc_curve(true_labels, prob_outputs)
        
        return results
    
    def plot_confusion_matrix(self, y_true, y_pred):
        """Karmaşıklık matrisini çizer"""
        from sklearn.metrics import confusion_matrix
        import seaborn as sns
        
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig('confusion_matrix_inference.png')
        plt.close()
    
    def plot_roc_curve(self, y_true, y_prob):
        """ROC eğrisini çizer"""
        from sklearn.metrics import roc_curve, auc
        
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.savefig('roc_curve_inference.png')
        plt.close()

def main():
    """Ana çalıştırma fonksiyonu"""
    # Excel dosyasını yükle
    try:
        data = pd.read_excel("YN_output.xlsx")
        print(f"Veri yüklendi. Boyut: {data.shape}")
    except Exception as e:
        print(f"Veri yükleme hatası: {e}")
        return
    
    # Girişleri ve çıkışları ayır
    if 'label' in data.columns:
        X = data.drop('label', axis=1).values
        y = data['label'].values
    else:
        # Son sütun etiket olarak varsayılır
        X = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values
    
    # Decoder sınıfını başlat
    input_size = X.shape[1]
    decoder = LDPCDecoder(model_path="ldpc_gru_model.pth", input_size=input_size)
    
    # Scaler'ı eğit
    decoder.fit_scaler(X)
    
    # Modeli test et
    results = decoder.batch_decode(X, y)
    
    # Örnek tahmin
    sample_idx = np.random.randint(0, len(X))
    sample_input = X[sample_idx:sample_idx+1]
    sample_true = y[sample_idx]
    
    pred, prob = decoder.decode(sample_input)
    
    print(f"\nÖrnek tahmin (indeks {sample_idx}):")
    print(f"  Gerçek değer: {sample_true}")
    print(f"  Tahmin: {pred[0]} (olasılık: {prob[0]:.4f})")
    
    # Sonuçları dışa aktar
    results_df = pd.DataFrame({
        'True_Label': y,
        'Predicted_Label': results['predictions'],
        'Probability': results['probabilities']
    })
    results_df.to_excel("gru_model_results.xlsx", index=False)
    print("\nSonuçlar 'gru_model_results.xlsx' dosyasına kaydedildi.")

if __name__ == "__main__":
    main()