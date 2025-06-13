import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import tqdm
from PIL import Image
import os

# НАСТРОЙКА УСТРОЙСТВА (GPU/CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используется устройство: {device}")

# ОПРЕДЕЛЕНИЕ ТРАНСФОРМАЦИЙ ДЛЯ ИЗОБРАЖЕНИЙ
# Для обучения применяем аугментацию данных (поворот, отражение, изменение яркости)
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),          # Изменяем размер до 224x224 пикселей
    transforms.RandomHorizontalFlip(p=0.5), # Случайное горизонтальное отражение
    transforms.RandomRotation(10),          # Случайный поворот до 10 градусов
    transforms.ColorJitter(brightness=0.2, contrast=0.2), # Изменение яркости и контрастности
    transforms.ToTensor(),                  # Преобразование в тензор
    transforms.Normalize(mean=[0.485, 0.456, 0.406],     # Нормализация по ImageNet
                        std=[0.229, 0.224, 0.225])
])

# Для валидации и тестирования не применяем аугментацию
val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

try:
    # Загружаем тренировочные данные
    full_train_data = datasets.ImageFolder(root='./data/CnD/train', transform=train_transform)
    
    # Разделяем на тренировочную и валидационную выборки (70% и 30%)
    train_size = int(0.7 * len(full_train_data))
    val_size = len(full_train_data) - train_size
    train_data, val_data = random_split(full_train_data, [train_size, val_size])
    
    # Применяем трансформации для валидации
    val_data.dataset.transform = val_test_transform
    
    # Загружаем тестовые данные
    test_data = datasets.ImageFolder(root='./data/CnD/test', transform=val_test_transform)
    
    print(f"Размер тренировочной выборки: {len(train_data)}")
    print(f"Размер валидационной выборки: {len(val_data)}")
    print(f"Размер тестовой выборки: {len(test_data)}")
    
except Exception as e:
    print(f"Ошибка загрузки данных: {e}")
    print("Убедитесь, что данные находятся в папке ./data/")

# СОЗДАНИЕ ЗАГРУЗЧИКОВ ДАННЫХ
train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=2)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False, num_workers=2)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False, num_workers=2)

# ФУНКЦИЯ ДЛЯ ВЫЧИСЛЕНИЯ ТОЧНОСТИ
def accuracy(pred, label):
    """
    Вычисляет точность предсказаний
    pred: тензор с логитами модели
    label: тензор с истинными метками
    """
    pred_labels = pred.argmax(dim=1)  # Получаем индекс максимального значения
    return (pred_labels == label).float().mean().item()

# ОПРЕДЕЛЕНИЕ АРХИТЕКТУРЫ CNN
class CatsDogsCNN(nn.Module):
    """
    Сверточная нейронная сеть для классификации кошек и собак
    Архитектура включает:
    - 3 сверточных блока с пулингом
    - 2 полносвязных слоя
    - Dropout для регуляризации
    """
    def __init__(self, num_classes=2):
        super(CatsDogsCNN, self).__init__()
        
        # ПЕРВЫЙ СВЕРТОЧНЫЙ БЛОК
        # Входные каналы: 3 (RGB), выходные: 32
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)  # 224x224x32
        self.bn1 = nn.BatchNorm2d(32)  # Батч-нормализация для стабилизации обучения
        self.pool1 = nn.MaxPool2d(2, 2)  # Пулинг 2x2, размер становится 112x112
        
        # ВТОРОЙ СВЕРТОЧНЫЙ БЛОК
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 112x112x64
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)  # Размер: 56x56
        
        # ТРЕТИЙ СВЕРТОЧНЫЙ БЛОК
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # 56x56x128
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)  # Размер: 28x28
        
        # ЧЕТВЕРТЫЙ СВЕРТОЧНЫЙ БЛОК
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)  # 28x28x256
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2, 2)  # Размер: 14x14
        
        # ГЛОБАЛЬНЫЙ УСРЕДНЯЮЩИЙ ПУЛИНГ
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  # Преобразует в 1x1x256
        
        # ПОЛНОСВЯЗНЫЕ СЛОИ
        self.fc1 = nn.Linear(256, 128)
        self.dropout1 = nn.Dropout(0.5)  # Dropout для предотвращения переобучения
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(64, num_classes)  # Выходной слой (2 класса)
        
        # ФУНКЦИЯ АКТИВАЦИИ
        self.relu = nn.ReLU()
    
    def forward(self, x):
        """
        Прямой проход через сеть
        """
        # Первый блок: свертка -> батч-норм -> ReLU -> пулинг
        x = self.pool1(self.relu(self.bn1(self.conv1(x))))
        
        # Второй блок
        x = self.pool2(self.relu(self.bn2(self.conv2(x))))
        
        # Третий блок
        x = self.pool3(self.relu(self.bn3(self.conv3(x))))
        
        # Четвертый блок
        x = self.pool4(self.relu(self.bn4(self.conv4(x))))
        
        # Глобальный пулинг
        x = self.global_pool(x)
        
        # Преобразование в одномерный вектор
        x = x.view(x.size(0), -1)  # Flatten
        
        # Полносвязные слои
        x = self.dropout1(self.relu(self.fc1(x)))
        x = self.dropout2(self.relu(self.fc2(x)))
        x = self.fc3(x)
        
        return x

# ФУНКЦИЯ ОБУЧЕНИЯ
def train_model(model, train_loader, val_loader, num_epochs=10, learning_rate=0.001):
    """
    Обучает модель
    """
    # Перемещаем модель на выбранное устройство
    model = model.to(device)
    
    # Определяем функцию потерь и оптимизатор
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # Планировщик обучения (уменьшает learning rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    # Списки для сохранения метрик
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    
    print("Начинаем обучение...")
    
    for epoch in range(num_epochs):
        # ЭТАП ОБУЧЕНИЯ
        model.train()  # Переводим модель в режим обучения
        train_loss = 0.0
        train_acc = 0.0
        
        # Прогресс-бар для тренировки
        train_pbar = tqdm.tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        
        for images, labels in train_pbar:
            # Перемещаем данные на устройство
            images, labels = images.to(device), labels.to(device)
            
            # Обнуляем градиенты
            optimizer.zero_grad()
            
            # Прямой проход
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Обратный проход
            loss.backward()
            optimizer.step()
            
            # Вычисляем метрики
            train_loss += loss.item()
            train_acc += accuracy(outputs, labels)
            
            # Обновляем прогресс-бар
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{accuracy(outputs, labels):.4f}'
            })
        
        # Средние значения за эпоху
        avg_train_loss = train_loss / len(train_loader)
        avg_train_acc = train_acc / len(train_loader)
        
        # ЭТАП ВАЛИДАЦИИ
        model.eval()  # Переводим модель в режим оценки
        val_loss = 0.0
        val_acc = 0.0
        
        with torch.no_grad():  # Отключаем вычисление градиентов
            val_pbar = tqdm.tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
            
            for images, labels in val_pbar:
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                val_acc += accuracy(outputs, labels)
                
                val_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{accuracy(outputs, labels):.4f}'
                })
        
        # Средние значения за эпоху
        avg_val_loss = val_loss / len(val_loader)
        avg_val_acc = val_acc / len(val_loader)
        
        # Сохраняем метрики
        train_losses.append(avg_train_loss)
        train_accuracies.append(avg_train_acc)
        val_losses.append(avg_val_loss)
        val_accuracies.append(avg_val_acc)
        
        # Обновляем learning rate
        scheduler.step()
        
        # Выводим результаты эпохи
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f}')
        print(f'  Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.4f}')
        print(f'  Learning Rate: {scheduler.get_last_lr()[0]:.6f}')
        print('-' * 50)
    
    return train_losses, train_accuracies, val_losses, val_accuracies

# ФУНКЦИЯ ТЕСТИРОВАНИЯ
def test_model(model, test_loader):
    """
    Тестирует модель на тестовой выборке
    """
    model.eval()
    test_loss = 0.0
    test_acc = 0.0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        test_pbar = tqdm.tqdm(test_loader, desc='Testing')
        
        for images, labels in test_pbar:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item()
            test_acc += accuracy(outputs, labels)
            
            test_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{accuracy(outputs, labels):.4f}'
            })
    
    avg_test_loss = test_loss / len(test_loader)
    avg_test_acc = test_acc / len(test_loader)
    
    print(f'Test Results:')
    print(f'  Test Loss: {avg_test_loss:.4f}')
    print(f'  Test Accuracy: {avg_test_acc:.4f}')
    
    return avg_test_loss, avg_test_acc

# ФУНКЦИЯ ДЛЯ ВИЗУАЛИЗАЦИИ РЕЗУЛЬТАТОВ
def plot_training_history(train_losses, train_accs, val_losses, val_accs):
    """
    Строит графики потерь и точности
    """
    plt.figure(figsize=(15, 5))
    
    # График потерь
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # График точности
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

# ОСНОВНАЯ ФУНКЦИЯ
if __name__ == "__main__":
    # Создаем модель
    model = CatsDogsCNN(num_classes=2)
    print(f"Модель создана. Количество параметров: {sum(p.numel() for p in model.parameters()):,}")
    
    # Обучаем модель (если данные загружены)
    try:
        # Обучение
        train_losses, train_accs, val_losses, val_accs = train_model(
            model, train_loader, val_loader, 
            num_epochs=8,  # Оптимальное количество эпох для предотвращения переобучения
            learning_rate=0.001
        )
        
        # Сохраняем модель
        torch.save(model.state_dict(), "cats_dogs_model.pt")
        print("Модель сохранена как 'cats_dogs_model.pt'")
        
        # Визуализируем результаты обучения
        plot_training_history(train_losses, train_accs, val_losses, val_accs)
        
        # Тестируем модель
        test_loss, test_acc = test_model(model, test_loader)
        
    except NameError:
        print("Данные не загружены. Загрузите датасет и повторите запуск.")

# ФУНКЦИЯ ДЛЯ ПРЕДСКАЗАНИЯ НА НОВЫХ ИЗОБРАЖЕНИЯХ
def predict_image(model, image_path, transform, class_names=['cat', 'dog']):
    """
    Делает предсказание для одного изображения
    """
    model.eval()
    
    # Загружаем и обрабатываем изображение
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)  # Добавляем batch dimension
    
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = F.softmax(output, dim=1)
        predicted_class = output.argmax(dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    return class_names[predicted_class], confidence

# Пример использования функции предсказания:
# prediction, confidence = predict_image(model, 'path/to/image.jpg', val_test_transform)
# print(f"Предсказание: {prediction}, Уверенность: {confidence:.4f}")