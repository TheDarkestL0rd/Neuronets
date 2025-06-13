import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import os
from CatsNDogs_Classificator import CatsDogsCNN
# Импортируем архитектуру модели из основного файла

class CatDogClassifierApp:
    """
    Графический интерфейс для классификации кошек и собак
    """
    def __init__(self, root):
        self.root = root
        self.root.title("Классификатор кошек и собак")
        self.root.geometry("800x600")
        self.root.configure(bg='#f0f0f0')
        
        # Инициализация переменных
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.current_image = None
        self.class_names = ['Кот', 'Собака']
        
        # Трансформации для предобработки изображений
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        self.setup_ui()
        self.load_model()
    
    def setup_ui(self):
        """
        Создание элементов интерфейса
        """
        # Заголовок
        title_label = tk.Label(
            self.root, 
            text="🐱 Классификатор кошек и собак 🐶",
            font=("Arial", 20, "bold"),
            bg='#f0f0f0',
            fg='#333333'
        )
        title_label.pack(pady=20)
        
        # Фрейм для кнопок
        button_frame = tk.Frame(self.root, bg='#f0f0f0')
        button_frame.pack(pady=10)
        
        # Кнопка загрузки изображения
        self.load_button = tk.Button(
            button_frame,
            text="📁 Выбрать изображение",
            command=self.load_image,
            font=("Arial", 12),
            bg='#4CAF50',
            fg='white',
            padx=20,
            pady=10,
            cursor='hand2'
        )
        self.load_button.pack(side=tk.LEFT, padx=10)
        
        # Кнопка классификации
        self.classify_button = tk.Button(
            button_frame,
            text="🔍 Классифицировать",
            command=self.classify_image,
            font=("Arial", 12),
            bg='#2196F3',
            fg='white',
            padx=20,
            pady=10,
            cursor='hand2',
            state='disabled'
        )
        self.classify_button.pack(side=tk.LEFT, padx=10)
        
        # Фрейм для отображения изображения
        self.image_frame = tk.Frame(self.root, bg='white', relief='sunken', bd=2)
        self.image_frame.pack(pady=20, padx=20, fill='both', expand=True)
        
        # Метка для отображения изображения
        self.image_label = tk.Label(
            self.image_frame,
            text="Выберите изображение для классификации",
            font=("Arial", 14),
            bg='white',
            fg='#666666'
        )
        self.image_label.pack(expand=True)
        
        # Фрейм для результатов
        self.result_frame = tk.Frame(self.root, bg='#f0f0f0')
        self.result_frame.pack(pady=20, padx=20, fill='x')
        
        # Метка для результата
        self.result_label = tk.Label(
            self.result_frame,
            text="",
            font=("Arial", 16, "bold"),
            bg='#f0f0f0'
        )
        self.result_label.pack()
        
        # Прогресс-бар для вероятностей
        self.prob_frame = tk.Frame(self.result_frame, bg='#f0f0f0')
        self.prob_frame.pack(pady=10, fill='x')
        
        # Статус бар
        self.status_label = tk.Label(
            self.root,
            text=f"Готов к работе | Устройство: {self.device}",
            font=("Arial", 10),
            bg='#e0e0e0',
            fg='#333333',
            relief='sunken'
        )
        self.status_label.pack(side=tk.BOTTOM, fill='x')
    
    def load_model(self):
        """
        Загрузка обученной модели
        """
        try:
            self.model = CatsDogsCNN(num_classes=2)
            
            # Проверяем наличие файла модели
            if os.path.exists("cats_dogs_model.pt"):
                self.model.load_state_dict(torch.load("cats_dogs_model.pt", map_location=self.device))
                self.model.to(self.device)
                self.model.eval()
                self.status_label.config(text=f"Модель загружена | Устройство: {self.device}")
            else:
                self.status_label.config(text="Модель не найдена! Обучите модель сначала.")
                messagebox.showwarning(
                    "Предупреждение", 
                    "Файл модели 'cats_dogs_model.pt' не найден!\n"
                    "Пожалуйста, сначала обучите модель, запустив основной скрипт."
                )
        except Exception as e:
            self.status_label.config(text=f"Ошибка загрузки модели: {str(e)}")
            messagebox.showerror("Ошибка", f"Не удалось загрузить модель: {str(e)}")
    
    def load_image(self):
        """
        Загрузка изображения через диалог выбора файла
        """
        file_types = [
            ("Изображения", "*.jpg *.jpeg *.png *.bmp *.gif *.tiff"),
            ("JPEG", "*.jpg *.jpeg"),
            ("PNG", "*.png"),
            ("Все файлы", "*.*")
        ]
        
        file_path = filedialog.askopenfilename(
            title="Выберите изображение",
            filetypes=file_types
        )
        
        if file_path:
            try:
                # Загружаем изображение
                image = Image.open(file_path).convert('RGB')
                self.current_image = image
                
                # Отображаем изображение в интерфейсе
                self.display_image(image)
                
                # Активируем кнопку классификации
                self.classify_button.config(state='normal')
                
                # Очищаем предыдущие результаты
                self.clear_results()
                
                self.status_label.config(text=f"Изображение загружено: {os.path.basename(file_path)}")
                
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось загрузить изображение: {str(e)}")
                self.status_label.config(text="Ошибка загрузки изображения")
    
    def display_image(self, image):
        """
        Отображение изображения в интерфейсе
        """
        # Изменяем размер для отображения
        display_size = (300, 300)
        image_resized = image.copy()
        image_resized.thumbnail(display_size, Image.Resampling.LANCZOS)
        
        # Конвертируем в формат для tkinter
        from PIL import ImageTk
        photo = ImageTk.PhotoImage(image_resized)
        
        # Обновляем метку изображения
        self.image_label.config(image=photo, text="")
        self.image_label.image = photo  # Сохраняем ссылку
    
    def classify_image(self):
        """
        Классификация загруженного изображения
        """
        if self.current_image is None:
            messagebox.showwarning("Предупреждение", "Сначала загрузите изображение!")
            return
        
        if self.model is None:
            messagebox.showerror("Ошибка", "Модель не загружена!")
            return
        
        try:
            self.status_label.config(text="Классификация...")
            self.root.update()
            
            # Предобработка изображения
            image_tensor = self.transform(self.current_image).unsqueeze(0).to(self.device)
            
            # Предсказание
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = F.softmax(outputs, dim=1)[0]
                predicted_class = outputs.argmax(dim=1).item()
                confidence = probabilities[predicted_class].item()
            
            # Отображение результатов
            self.display_results(predicted_class, probabilities)
            
            self.status_label.config(
                text=f"Классификация завершена: {self.class_names[predicted_class]} ({confidence:.2%})"
            )
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при классификации: {str(e)}")
            self.status_label.config(text="Ошибка классификации")
    
    def display_results(self, predicted_class, probabilities):
        """
        Отображение результатов классификации
        """
        # Основной результат
        confidence = probabilities[predicted_class].item()
        result_text = f"Результат: {self.class_names[predicted_class]}"
        confidence_text = f"Уверенность: {confidence:.2%}"
        
        # Цвет результата
        if confidence > 0.8:
            color = '#4CAF50'  # Зеленый
        elif confidence > 0.6:
            color = '#FF9800'  # Оранжевый
        else:
            color = '#F44336'  # Красный
        
        self.result_label.config(
            text=f"{result_text}\n{confidence_text}",
            fg=color
        )
        
        # Очищаем предыдущие прогресс-бары
        for widget in self.prob_frame.winfo_children():
            widget.destroy()
        
        # Создаем прогресс-бары для каждого класса
        for i, (class_name, prob) in enumerate(zip(self.class_names, probabilities)):
            # Рамка для каждого класса
            class_frame = tk.Frame(self.prob_frame, bg='#f0f0f0')
            class_frame.pack(fill='x', pady=5)
            
            # Название класса
            class_label = tk.Label(
                class_frame,
                text=f"{class_name}:",
                font=("Arial", 12),
                bg='#f0f0f0',
                width=10,
                anchor='w'
            )
            class_label.pack(side='left')
            
            # Прогресс-бар
            progress = ttk.Progressbar(
                class_frame,
                length=200,
                value=prob.item() * 100,
                mode='determinate'
            )
            progress.pack(side='left', padx=10)
            
            # Процент
            percent_label = tk.Label(
                class_frame,
                text=f"{prob.item():.1%}",
                font=("Arial", 10),
                bg='#f0f0f0'
            )
            percent_label.pack(side='left', padx=5)
    
    def clear_results(self):
        """
        Очистка результатов классификации
        """
        self.result_label.config(text="")
        for widget in self.prob_frame.winfo_children():
            widget.destroy()

def main():
    """
    Главная функция для запуска приложения
    """
    # Создаем главное окно
    root = tk.Tk()
    
    # Устанавливаем иконку (если есть)
    try:
        root.iconbitmap('icon.ico')  # Опционально
    except:
        pass
    
    # Создаем приложение
    app = CatDogClassifierApp(root)
    
    # Запускаем главный цикл
    root.mainloop()

if __name__ == "__main__":
    main()