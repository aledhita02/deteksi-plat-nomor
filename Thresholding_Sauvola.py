import cv2
import numpy as np
from skimage.filters import threshold_sauvola
import os

#Pada tapa ini didefinisikan Path folder tempat gambar berada
input_folder = "./input_100"  # folder gambar asli
output_folder = "./output_100"  # Folder untuk hasil thresholding sauvola

# Membuat folder output jika belum ada
os.makedirs(output_folder, exist_ok=True)

# Parameter Sauvola
window_size = 15  # Ukuran jendela
k = 0.2           # Nilai k

# Inisialisasi nomor untuk penamaan gambarb agar lebih mudah
image_number = 1

# Memproses setiap gambar dalam folder
for file_name in os.listdir(input_folder):
    if file_name.endswith(('.jpg', '.jpeg', '.png')):  
        # Membuat nama input dan output berdasarkan nomor
        input_renamed = f"gambar_{image_number}.jpg"
        output_renamed = f"gambar_filtering_{image_number}.jpg"
        
        input_path = os.path.join(input_folder, file_name)
        renamed_input_path = os.path.join(input_folder, input_renamed)
        output_path = os.path.join(output_folder, output_renamed)

        # Ubah nama file input menjadi "gambar_nomor"
        os.rename(input_path, renamed_input_path)

        # Membaca citra
        image = cv2.imread(renamed_input_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Error loading {input_renamed}. Skipping...")
            continue

        # Menghitung threshold Sauvola
        threshold = threshold_sauvola(image, window_size=window_size, k=k)

        # melakukan binarisasi pada citra
        binary_image = (image > threshold).astype(np.uint8) * 255

        # Menyimpan hasil
        cv2.imwrite(output_path, binary_image)
        print(f"Processed: {input_renamed} -> Saved: {output_renamed}")

        # Increment nomor
        image_number += 1

print("Proses Selesai")


