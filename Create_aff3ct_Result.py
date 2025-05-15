#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

def create_sample_aff3ct_results(output_file="aff3ct_results.csv"):
    """AFF3CT sonuçları için örnek bir CSV dosyası oluşturur"""
    
    # SNR değerleri (dB)
    snr_values = np.arange(0, 5.5, 0.5)
    
    # Farklı SNR değerleri için farklı tekrarlar
    repetitions = 10
    
    # Sonuç dataframe'i için veriler
    data = []
    
    for snr in snr_values:
        # BER ve FER değerlerini SNR'ye göre hesapla (gerçekçi bir model)
        # SNR arttıkça BER/FER değerleri üstel olarak azalır
        base_ber = 0.5 * np.exp(-snr)  # Temel BER değeri
        base_fer = min(1.0, base_ber * 10)  # FER genellikle BER'den daha yüksektir
        
        # Her SNR için birden çok ölçüm
        for i in range(repetitions):
            # Rastgele değişim ekle
            random_factor = np.random.uniform(0.9, 1.1)
            ber = base_ber * random_factor
            fer = base_fer * random_factor
            
            # Frame uzunluğu ve bit sayısı (örnek değerler)
            frame_size = 2048
            n_frames = 1000
            be_count = int(ber * frame_size * n_frames)
            fe_count = int(fer * n_frames)
            
            # Süre (SNR düşükken daha fazla iterasyon gerekebilir)
            time_sec = 10.0 * (1.0 + np.exp(-snr/2))
            
            # Satır ekle
            data.append({
                'SNR': snr,
                'BER': ber,
                'FER': fer,
                'BE count': be_count,
                'FE count': fe_count,
                'N_frames': n_frames,
                'Time (s)': time_sec
            })
    
    # DataFrame oluştur
    df = pd.DataFrame(data)
    
    # CSV dosyasına kaydet
    df.to_csv(output_file, index=False)
    
    print(f"Örnek AFF3CT sonuçları oluşturuldu ve '{output_file}' dosyasına kaydedildi.")
    print(f"Toplam {len(df)} satır veri içeriyor.")
    
    return df

if __name__ == "__main__":
    create_sample_aff3ct_results()