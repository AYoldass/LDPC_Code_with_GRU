#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from time import time
import torch

# Kendi modüllerimizi içe aktar
from LDPC_GRU_Training_Model import main as train_model
from Training_Dataset_and_Model_Test import prepare_data_from_excel, analyze_data_distribution, test_trained_model
from LDPC_GRU_Model_Inference import LDPCDecoder
from LDPC_GRU_and_Normal_LDPC_Comp import evaluate_aff3ct_performance, evaluate_gru_model_by_snr, plot_comparison, benchmark_performance

def parse_arguments():
    """Komut satırı argümanlarını ayrıştırır"""
    parser = argparse.ArgumentParser(description='LDPC GRU Sistemi')
    parser.add_argument('--mode', type=str, default='all', choices=['all', 'prepare', 'train', 'test', 'inference', 'compare'],
                      help='Çalıştırma modu (all, prepare, train, test, inference, compare)')
    parser.add_argument('--data', type=str, default='data1_csv',
                      help='Veri seti dosya yolu')
    parser.add_argument('--model', type=str, default='ldpc_gru_model.pth',
                      help='Model dosya yolu (kaydetme veya yükleme için)')
    parser.add_argument('--epochs', type=int, default=100,
                      help='Eğitim epoch sayısı')
    parser.add_argument('--batch_size', type=int, default=64,
                      help='Batch boyutu')
    parser.add_argument('--aff3ct_results', type=str, default='aff3ct_results.csv',
                      help='AFF3CT sonuç dosyası')
    parser.add_argument('--save_dir', type=str, default='results',
                      help='Sonuçları kaydetme dizini')
    parser.add_argument('--debug', action='store_true',
                      help='Debug modu (daha fazla log)')
    
    return parser.parse_args()

def create_results_dir(directory):
    """Sonuçları kaydetmek için dizin oluşturur"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"'{directory}' dizini oluşturuldu.")
    else:
        print(f"'{directory}' dizini zaten mevcut.")

def prepare_data(args):
    """Veri setini hazırlar ve analiz eder"""
    print("="*80)
    print("VERİ HAZIRLAMA VE ANALİZ")
    print("="*80)
    
    data = prepare_data_from_excel(args.data)
    
    if data is not None:
        # Veri analizi
        print("\nVeri dağılımı analiz ediliyor...")
        analyze_data_distribution(data)
        
        # Veri hakkında özet bilgi
        print("\nVeri özeti:")
        print(f"- Toplam satır sayısı: {data.shape[0]}")
        print(f"- Toplam özellik sayısı: {data.shape[1]}")
        
        # Etiket dağılımı (son sütun etiket kabul ediliyor)
        if 'label' in data.columns:
            label_col = 'label'
        else:
            label_col = data.columns[-1]
        
        label_counts = data[label_col].value_counts()
        print(f"\nEtiket dağılımı ({label_col}):")
        for label, count in label_counts.items():
            print(f"- {label}: {count} ({count/len(data)*100:.2f}%)")
        
        return data
    else:
        print("Veri hazırlama hatası!")
        return None

def train_gru_model(args):
    """GRU modelini eğitir"""
    print("="*80)
    print("GRU MODEL EĞİTİMİ")
    print("="*80)
    
    # Eğitim fonksiyonunu çağır
    train_model(args.data)
    
    # Model dosyasını kontrol et
    if os.path.exists(args.model):
        print(f"Model başarıyla kaydedildi: {args.model}")
        return True
    else:
        print(f"Model kaydetme hatası! Dosya bulunamadı: {args.model}")
        return False

def test_gru_model(args):
    """Eğitilmiş modeli test eder"""
    print("="*80)
    print("GRU MODEL TESTİ")
    print("="*80)
    
    # Test verilerini hazırla
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from torch.utils.data import DataLoader
    from LDPC_GRU_Training_Model import LDPCDataset
    
    # Veriyi yükle
    data = pd.read_excel(args.data)
    
    # Girişleri ve etiketleri ayır
    if 'label' in data.columns:
        X = data.drop('label', axis=1).values
        y = data['label'].values
    else:
        X = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values
    
    # Veriyi bölme
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Veriyi ölçeklendirme
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Test veri kümesini oluştur
    test_dataset = LDPCDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    # Cihaz seçimi
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Kullanılan cihaz: {device}")
    
    # Modeli test et
    accuracies = []
    predictions = []
    true_labels = []
    
    for _ in range(3):  # Ortalama sonuç için 3 kez çalıştır
        preds, labels, acc = test_trained_model(args.model, test_loader, device)
        accuracies.append(acc)
        predictions.extend(preds)
        true_labels.extend(labels)
    
    avg_accuracy = sum(accuracies) / len(accuracies)
    print(f"\nOrtalama test doğruluğu: {avg_accuracy:.2f}%")
    
    return avg_accuracy, predictions, true_labels

def run_inference(args):
    """Model çıkarımı yapar"""
    print("="*80)
    print("MODEL ÇIKARIMI")
    print("="*80)
    
    # Veriyi yükle
    data = pd.read_excel(args.data)
    
    # Girişleri ve etiketleri ayır
    if 'label' in data.columns:
        X = data.drop('label', axis=1).values
        y = data['label'].values
    else:
        X = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values
    
    # Decoder sınıfını başlat
    input_size = X.shape[1]
    decoder = LDPCDecoder(model_path=args.model, input_size=input_size)
    
    # Scaler'ı eğit
    decoder.fit_scaler(X)
    
    # Modeli test et
    print("\nTüm test verileri için çıkarım yapılıyor...")
    results = decoder.batch_decode(X, y)
    
    # Sonuçları kaydet
    results_df = pd.DataFrame({
        'True_Label': y,
        'Predicted_Label': results['predictions'],
        'Probability': results['probabilities']
    })
    
    # SNR değerini ekle (eğer varsa)
    if 'SNR' in data.columns:
        results_df['SNR'] = data['SNR'].values
    
    results_file = os.path.join(args.save_dir, "gru_model_results.xlsx")
    results_df.to_excel(results_file, index=False)
    print(f"Sonuçlar '{results_file}' dosyasına kaydedildi.")
    
    return results

def compare_with_aff3ct(args):
    """GRU modeli ile AFF3CT karşılaştırması yapar"""
    print("="*80)
    print("KARŞILAŞTIRMA: GRU vs AFF3CT")
    print("="*80)
    
    # AFF3CT sonuçlarını değerlendir
    print("\nAFF3CT sonuçları değerlendiriliyor...")
    aff3ct_results = evaluate_aff3ct_performance(args.aff3ct_results)
    
    # GRU sonuçlarını değerlendir
    gru_results_file = os.path.join(args.save_dir, "gru_model_results.xlsx")
    print(f"\nGRU sonuçları değerlendiriliyor: {gru_results_file}")
    
    # Dosya var mı kontrol et
    if not os.path.exists(gru_results_file):
        print(f"Uyarı: GRU sonuç dosyası bulunamadı! Önce çıkarım yapmanız gerekebilir.")
        return None
    
    # SNR sütun adını kontrol et
    snr_col = "SNR"
    gru_results = evaluate_gru_model_by_snr(gru_results_file, snr_col)
    
    # Karşılaştırma grafiklerini çiz
    comparison_results = plot_comparison(aff3ct_results, gru_results, args.save_dir)
    
    # Çalışma zamanı karşılaştırması yap
    print("\nÇalışma zamanı karşılaştırması yapılıyor...")
    perf_results = benchmark_performance(args.model, args.data, None)
    
    # Karşılaştırma sonuçlarını dışa aktar
    results_df = pd.DataFrame({
        'SNR': comparison_results['snr'],
        'AFF3CT_BER': comparison_results['aff3ct_ber'],
        'AFF3CT_FER': comparison_results['aff3ct_fer'],
        'GRU_BER': comparison_results['gru_ber'],
        'GRU_FER': comparison_results['gru_fer'],
        'GRU_Accuracy': comparison_results['gru_accuracy']
    })
    
    results_file = os.path.join(args.save_dir, "comparison_results.xlsx")
    results_df.to_excel(results_file, index=False)
    print(f"Karşılaştırma sonuçları '{results_file}' dosyasına kaydedildi.")
    
    return comparison_results

def main():
    """Ana çalıştırma fonksiyonu"""
    # Argümanları ayrıştır
    args = parse_arguments()
    
    # Sonuçları kaydetmek için dizin oluştur
    create_results_dir(args.save_dir)
    
    # Debug modu
    if args.debug:
        print("Debug modu açık!")
        print(f"Argümanlar: {args}")
    
    # Seçilen moda göre işlem yap
    if args.mode in ['all', 'prepare']:
        data = prepare_data(args)
    
    if args.mode in ['all', 'train']:
        train_success = train_gru_model(args)
    
    if args.mode in ['all', 'test']:
        if os.path.exists(args.model):
            acc, preds, labels = test_gru_model(args)
        else:
            print(f"Hata: Model dosyası bulunamadı: {args.model}")
            print("Önce modeli eğitmeniz gerekiyor!")
    
    if args.mode in ['all', 'inference']:
        if os.path.exists(args.model):
            results = run_inference(args)
        else:
            print(f"Hata: Model dosyası bulunamadı: {args.model}")
            print("Önce modeli eğitmeniz gerekiyor!")
    
    if args.mode in ['all', 'compare']:
        comp_results = compare_with_aff3ct(args)
    
    print("\nTüm işlemler tamamlandı!")

if __name__ == "__main__":
    main()