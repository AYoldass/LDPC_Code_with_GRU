#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import special

def generate_awgn_samples(snr_db, num_samples=1000, input_bits=None):
    """
    Verilen SNR değerinde AWGN (Additive White Gaussian Noise) örnekleri oluşturur.
    
    Args:
        snr_db (float): Sinyal gürültü oranı (dB)
        num_samples (int): Oluşturulacak örnek sayısı
        input_bits (numpy array): Giriş bitleri (None ise rastgele üretilir)
    
    Returns:
        input_bits (numpy array): Giriş bitleri
        received_signal (numpy array): Alınan sinyal
        noise (numpy array): Gürültü
    """
    # SNR değerini lineer forma dönüştür
    snr_linear = 10 ** (snr_db / 10)
    
    # Giriş bitleri yoksa rastgele oluştur
    if input_bits is None:
        input_bits = np.random.randint(0, 2, num_samples)
    
    # BPSK modülasyonu (+1/-1)
    modulated_signal = 2 * input_bits - 1
    
    # Gürültü güç seviyesini hesapla (sinyal gücü 1 kabul edilerek)
    noise_power = 1 / snr_linear
    noise_std = np.sqrt(noise_power)
    
    # AWGN gürültüsü ekle
    noise = np.random.normal(0, noise_std, len(modulated_signal))
    received_signal = modulated_signal + noise
    
    return input_bits, received_signal, noise

def calculate_llr(received_signal, noise_var):
    """
    Alınan sinyalden LLR (Log-Likelihood Ratio) değerlerini hesaplar.
    
    Args:
        received_signal (numpy array): Alınan sinyal
        noise_var (float): Gürültü varyansı
    
    Returns:
        llr (numpy array): LLR değerleri
    """
    return (2 / noise_var) * received_signal

def generate_ldpc_test_dataset(snr_values, samples_per_snr=1000, feature_size=24, frame_size=2048):
    """
    LDPC test veri seti oluşturur.
    
    Args:
        snr_values (list): SNR değerleri (dB)
        samples_per_snr (int): Her SNR değeri için örnek sayısı
        feature_size (int): Her satır için özellik sayısı
        frame_size (int): Her çerçevedeki bit sayısı
    
    Returns:
        pandas DataFrame: Oluşturulan veri seti
    """
    data_list = []
    
    for snr in snr_values:
        print(f"SNR = {snr} dB için veri oluşturuluyor...")
        
        # SNR değerine göre gürültü varyansını hesapla
        snr_linear = 10 ** (snr / 10)
        noise_var = 1 / snr_linear
        
        for i in range(samples_per_snr):
            # Gerçek bitleri oluştur (0 veya 1)
            input_bits = np.random.randint(0, 2, feature_size - 1)
            
            # AWGN üzerinden iletim simüle et
            _, received_signal, _ = generate_awgn_samples(snr, num_samples=feature_size-1, input_bits=input_bits)
            
            # LLR değerlerini hesapla
            llr_values = calculate_llr(received_signal, noise_var)
            
            # Örnek hata yapma olasılığını hesapla (SNR'ye bağlı)
            error_prob = 0.5 * special.erfc(np.sqrt(snr_linear))
            
            # Hard Decision sonucu
            hard_decisions = (received_signal > 0).astype(int)
            
            # Hatalar
            errors = (hard_decisions != input_bits).astype(int)
            
            # Bit hata olup olmadığını belirle (label)
            has_error = 1 if np.sum(errors) > 0 else 0
            
            # Veri satırı oluştur
            row_data = np.concatenate([llr_values, [has_error]])
            
            # Dictionary olarak ekle
            feature_dict = {f'feature_{j}': val for j, val in enumerate(llr_values)}
            feature_dict['SNR'] = snr
            feature_dict['label'] = has_error
            
            data_list.append(feature_dict)
    
    # DataFrame oluştur
    df = pd.DataFrame(data_list)
    
    return df

def main():
    # Test için SNR değerleri
    snr_values = np.arange(0, 5.5, 0.5)
    
    # Her SNR değeri için örnek sayısı
    samples_per_snr = 200
    
    # Özellik sayısı - gerçekçi bir LDPC decoder için LLR değerleri
    feature_size = 24
    
    # Veri setini oluştur
    df = generate_ldpc_test_dataset(snr_values, samples_per_snr, feature_size)
    
    # Özet bilgi
    print("\nOluşturulan veri seti:")
    print(f"Toplam örnek sayısı: {len(df)}")
    print(f"Sütunlar: {df.columns.tolist()}")
    print("\nİlk birkaç satır:")
    print(df.head())
    
    # Etiket dağılımı
    label_counts = df['label'].value_counts()
    print("\nEtiket dağılımı:")
    print(label_counts)
    print(f"Pozitif örnekler: {label_counts[1]} ({label_counts[1]/len(df)*100:.2f}%)")
    print(f"Negatif örnekler: {label_counts[0]} ({label_counts[0]/len(df)*100:.2f}%)")
    
    # SNR'ye göre hata dağılımını göster
    plt.figure(figsize=(10, 6))
    error_by_snr = df.groupby('SNR')['label'].mean()
    plt.plot(error_by_snr.index, error_by_snr.values, 'o-')
    plt.grid(True)
    plt.xlabel('SNR (dB)')
    plt.ylabel('Hata Oranı')
    plt.title('SNR Değerine Göre Hata Oranı')
    plt.savefig('error_by_snr.png')
    plt.close()
    
    # Excel dosyasına kaydet
    output_file = "ldpc_test_data.xlsx"
    df.to_excel(output_file, index=False)
    print(f"\nVeri seti '{output_file}' dosyasına kaydedildi.")

if __name__ == "__main__":
    main()