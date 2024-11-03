# iris_classifier_gui.py

import tkinter as tk
from tkinter import messagebox
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Memuat dataset Iris
iris = datasets.load_iris()
X = iris.data  # Fitur: sepal dan petal length/width
y = iris.target  # Target: spesies

# Membagi data menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Membangun model KNN
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# Fungsi untuk mengklasifikasikan spesies iris
def classify_iris(sepal_length, sepal_width, petal_length, petal_width):
    input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
    predicted_class = model.predict(input_data)
    return iris.target_names[predicted_class][0]

# Fungsi untuk menangani klik tombol
def on_classify():
    try:
        sepal_length = float(entry_sepal_length.get())
        sepal_width = float(entry_sepal_width.get())
        petal_length = float(entry_petal_length.get())
        petal_width = float(entry_petal_width.get())
        
        predicted_species = classify_iris(sepal_length, sepal_width, petal_length, petal_width)
        messagebox.showinfo("Hasil Klasifikasi", f'Spesies yang diprediksi: {predicted_species}')
    except ValueError:
        messagebox.showerror("Input Error", "Mohon masukkan angka yang valid.")

# Membuat jendela utama
root = tk.Tk()
root.title("Aplikasi Klasifikasi Iris")

# Label dan Entry untuk panjang sepal
tk.Label(root, text="Panjang Sepal (cm):").grid(row=0, column=0)
entry_sepal_length = tk.Entry(root)
entry_sepal_length.grid(row=0, column=1)

# Label dan Entry untuk lebar sepal
tk.Label(root, text="Lebar Sepal (cm):").grid(row=1, column=0)
entry_sepal_width = tk.Entry(root)
entry_sepal_width.grid(row=1, column=1)

# Label dan Entry untuk panjang petal
tk.Label(root, text="Panjang Petal (cm):").grid(row=2, column=0)
entry_petal_length = tk.Entry(root)
entry_petal_length.grid(row=2, column=1)

# Label dan Entry untuk lebar petal
tk.Label(root, text="Lebar Petal (cm):").grid(row=3, column=0)
entry_petal_width = tk.Entry(root)
entry_petal_width.grid(row=3, column=1)

# Tombol untuk klasifikasi
button_classify = tk.Button(root, text="Klasifikasikan", command=on_classify)
button_classify.grid(row=4, column=0, columnspan=2)

# Menjalankan aplikasi
root.mainloop()
