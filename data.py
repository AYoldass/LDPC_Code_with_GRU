import pandas as pd

def find_problematic_blocks(input_filename):
    """
    Satır satır okuyarak blok blok analiz eder ve problemi tam olarak tespit eder.
    """
    print("Dosya okunuyor:", input_filename)
    
    # Beklenen blok boyutu
    expected_block_size = 647
    
    # Blok bilgilerini saklamak için listeler
    all_decoder_blocks = []
    all_encoder_blocks = []
    problematic_blocks = []
    
    with open(input_filename, 'r') as file:
        lines = file.readlines()
    
    # Blok takibi için değişkenler
    current_block = -1
    current_decoder_values = []
    current_encoder_values = []
    in_decoder_section = False
    in_encoder_section = False
    decoder_indices_seen = set()
    encoder_indices_seen = set()
    
    for line_idx, line in enumerate(lines):
        line_num = line_idx + 1
        line = line.strip()
        
        # Yeni blok başlangıcını kontrol et (Decoder input Y_N[0])
        if "Decoder input Y_N[0]" in line:
            # Önceki bloğu tamamla ve kontrol et
            if current_block >= 0:
                if len(current_decoder_values) != expected_block_size or len(current_encoder_values) != expected_block_size:
                    print(f"\nPROBLEMLİ BLOK BULUNDU: Blok {current_block}")
                    print(f"  Decoder değerleri: {len(current_decoder_values)} (beklenen {expected_block_size})")
                    print(f"  Encoder değerleri: {len(current_encoder_values)} (beklenen {expected_block_size})")
                    
                    # Hangi indekslerin eksik olduğunu bul
                    expected_indices = set(range(expected_block_size))
                    missing_decoder = sorted(expected_indices - decoder_indices_seen)
                    missing_encoder = sorted(expected_indices - encoder_indices_seen)
                    
                    if missing_decoder:
                        print(f"  Eksik decoder indeksleri: {missing_decoder}")
                    if missing_encoder:
                        print(f"  Eksik encoder indeksleri: {missing_encoder}")
                    
                    problematic_blocks.append({
                        "block": current_block,
                        "decoder_count": len(current_decoder_values),
                        "encoder_count": len(current_encoder_values),
                        "missing_decoder": missing_decoder,
                        "missing_encoder": missing_encoder
                    })
                else:
                    # Blok tam ise listeye ekle
                    all_decoder_blocks.append(current_decoder_values.copy())
                    all_encoder_blocks.append(current_encoder_values.copy())
                    
            # Yeni blok başlat
            current_block += 1
            current_decoder_values = []
            current_encoder_values = []
            decoder_indices_seen = set()
            encoder_indices_seen = set()
            in_decoder_section = True
            in_encoder_section = False
            
            # İlk satırın değerini işle
            value = float(line.split("=")[1].strip())
            current_decoder_values.append(value)
            decoder_indices_seen.add(0)
        
        # Diğer decoder satırlarını işle
        elif "Decoder input Y_N[" in line:
            # İndeksi çıkart
            idx_start = line.find("[") + 1
            idx_end = line.find("]")
            index = int(line[idx_start:idx_end])
            
            # Değeri çıkart
            value = float(line.split("=")[1].strip())
            
            # Değeri ve indeksi kaydet
            if len(current_decoder_values) < expected_block_size:
                current_decoder_values.append(value)
                decoder_indices_seen.add(index)
            
            in_decoder_section = True
            # Decoder'dan encoder'a geçiş olup olmadığını kontrol et
            if index == expected_block_size - 1:
                in_decoder_section = False
        
        # Encoder satırlarını işle  
        elif "Encoder output index" in line:
            # İndeksi çıkart
            index = int(line.split("index")[1].split("->")[0].strip())
            
            # Değeri çıkart  
            value = int(line.split("->")[1].strip())
            
            # Değeri ve indeksi kaydet
            if len(current_encoder_values) < expected_block_size:
                current_encoder_values.append(value)
                encoder_indices_seen.add(index)
            
            in_encoder_section = True
            in_decoder_section = False
    
    # Son bloğu kontrol et
    if len(current_decoder_values) != expected_block_size or len(current_encoder_values) != expected_block_size:
        print(f"\nPROBLEMLİ BLOK BULUNDU: Blok {current_block}")
        print(f"  Decoder değerleri: {len(current_decoder_values)} (beklenen {expected_block_size})")
        print(f"  Encoder değerleri: {len(current_encoder_values)} (beklenen {expected_block_size})")
        
        # Hangi indekslerin eksik olduğunu bul
        expected_indices = set(range(expected_block_size))
        missing_decoder = sorted(expected_indices - decoder_indices_seen)
        missing_encoder = sorted(expected_indices - encoder_indices_seen)
        
        if missing_decoder:
            print(f"  Eksik decoder indeksleri: {missing_decoder}")
        if missing_encoder:
            print(f"  Eksik encoder indeksleri: {missing_encoder}")
        
        problematic_blocks.append({
            "block": current_block,
            "decoder_count": len(current_decoder_values),
            "encoder_count": len(current_encoder_values),
            "missing_decoder": missing_decoder,
            "missing_encoder": missing_encoder
        })
    else:
        # Son blok tam ise listeye ekle
        all_decoder_blocks.append(current_decoder_values.copy())
        all_encoder_blocks.append(current_encoder_values.copy())
    
    # Özet
    total_blocks = current_block + 1
    print(f"\nÖzet:")
    print(f"Toplam blok sayısı: {total_blocks}")
    print(f"Tam blok sayısı: {total_blocks - len(problematic_blocks)}")
    print(f"Problemli blok sayısı: {len(problematic_blocks)}")
    
    if problematic_blocks:
        print("\nProblemli blokların detayları:")
        for block in problematic_blocks:
            print(f"Blok {block['block']}:")
            print(f"  Decoder değerleri: {block['decoder_count']} (beklenen {expected_block_size})")
            print(f"  Encoder değerleri: {block['encoder_count']} (beklenen {expected_block_size})")
            if block['missing_decoder']:
                print(f"  Eksik decoder indeksleri: {block['missing_decoder'][:10]}{'...' if len(block['missing_decoder']) > 10 else ''}")
            if block['missing_encoder']:
                print(f"  Eksik encoder indeksleri: {block['missing_encoder'][:10]}{'...' if len(block['missing_encoder']) > 10 else ''}")
    
    return all_decoder_blocks, all_encoder_blocks, problematic_blocks

def create_dataframe_from_blocks(decoder_blocks, encoder_blocks, output_filename):
    """
    Tam bloklardan DataFrame oluştur ve CSV olarak kaydet.
    """
    # DataFrame oluştur
    data = {
        'Decoder': decoder_blocks,
        'Encoder': encoder_blocks
    }
    
    df = pd.DataFrame(data)
    print(f"{len(df)} tam blokla DataFrame oluşturuluyor")
    print(f"{output_filename} dosyasına kaydediliyor")
    df.to_csv(output_filename, index=False)
    
    return len(df)

# Ana kod
if __name__ == "__main__":
    # Dosya yolları
    input_file = "/home/ayoldass/Masaüstü/aff3ct_gru/dataset/data.txt"
    output_file = "data1.csv"
    
    # Problemli blokları bul ve geçerli blokları al
    decoder_blocks, encoder_blocks, problematic_blocks = find_problematic_blocks(input_file)
    
    # DataFrame oluştur ve kaydet
    if decoder_blocks and encoder_blocks:
        create_dataframe_from_blocks(decoder_blocks, encoder_blocks, output_file)
    else:
        print("Kaydedilecek tam blok bulunamadı.")