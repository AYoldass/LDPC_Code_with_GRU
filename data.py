import re
import csv
import pandas as pd
from rich import print
def parse_input_file(input_filename, output_filename):
    # Initialize dictionaries to store the values by their indices
    data ={}
    decoder = []
    encoder = []
    
    # Regular expressions to match only the specific patterns
    decoder_pattern = r'Decoder input Y_N\[(\d+)\] = ([-+]?\d*\.\d+|\d+)'
    encoder_pattern = r'Encoder output index (\d+) -> (\d+)'
    print("Decoder pattern:", decoder_pattern)
    print("Encoder pattern:", encoder_pattern)
    
    # # Read the input file
    with open(input_filename, 'r') as file:
        content = file.read()
        
        # Extract decoder values - only from "Decoder input" lines
        decoder_matches = re.findall(decoder_pattern, content)
        encoder_matches = re.findall(encoder_pattern, content)

        for index, value in decoder_matches:
            decoder.append(float(value))
        print(len(decoder))
        
        for index, value in encoder_matches:
            encoder.append(int(value))
        data['Decoder'] = [decoder[i:i+540] for i in range(0,len(decoder),540)]
        data['Encoder'] = [encoder[i:i+540] for i in range(0,len(encoder),540)]
        print(data["Decoder"]),
        pd.DataFrame(data).to_csv(output_filename, index=False)
            
        
            
     

# Example usage
if __name__ == "__main__":
    input_filename = "data1.txt"  # Replace with your input file name
    output_filename = "data1.csv"  # Replace with your desired output file name
    parse_input_file(input_filename, output_filename)

