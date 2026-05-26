import torch
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
from PIL import Image
import os

class DogBreedClassifier:
    def __init__(self, model_path: str, breed_names: list):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.breed_names = breed_names
        print(f"✅ Загружено {len(self.breed_names)} названий пород")
        print(f"🔧 Инициализация классификатора на устройстве: {self.device}")
        
        # Загружаем архитектуру EfficientNet-B4 (НЕ B0!)
        print("📦 Создание архитектуры EfficientNet-B4...")
        self.model = efficientnet_b4(weights=None)
        
        # Заменяем классификатор для 120 пород (как при обучении)
        num_classes = 120
        self.model.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=0.3, inplace=True),
            torch.nn.Linear(1792, 1024),  # B4 имеет 1792 выходных признака
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.4),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.3),
            torch.nn.Linear(512, num_classes)
        )
        
        # Загружаем веса
        print(f"📂 Загрузка весов из {model_path}...")
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=self.device)
            
            # Удаляем лишние ключи, если есть
            new_state_dict = {}
            for k, v in state_dict.items():
                if 'module.' in k:
                    k = k.replace('module.', '')
                new_state_dict[k] = v
            
            # Загружаем с строгим соответствием (strict=True)
            self.model.load_state_dict(new_state_dict, strict=True)
            print(f"✅ Модель загружена из {model_path}")
        else:
            print(f"❌ Ошибка: файл {model_path} не найден!")
            raise FileNotFoundError(f"Модель не найдена: {model_path}")
        
        self.model.to(self.device)
        self.model.eval()
        print(f"✅ Модель переведена в режим оценки на {self.device}")
        
        # Трансформации для B4 (размер 380)
        self.transform = transforms.Compose([
            transforms.Resize(400),      # Немного больше для лучшего качества
            transforms.CenterCrop(380),  # B4 использует 380x380
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        print("✅ Трансформации настроены")
    
    def predict(self, image_path: str, top_k=3):
        """Возвращает топ-k пород с процентами уверенности."""
        print(f"\n📷 Классификация изображения: {image_path}")
        
        try:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Файл не найден: {image_path}")
            
            file_size = os.path.getsize(image_path) / 1024
            print(f"   Размер файла: {file_size:.1f} КБ")
            
            image = Image.open(image_path).convert('RGB')
            print(f"   Оригинальный размер: {image.size}")
            
            input_tensor = self.transform(image)
            print(f"   После трансформаций: {input_tensor.shape}")
            
            input_batch = input_tensor.unsqueeze(0).to(self.device)
            print(f"   Batch тензор: {input_batch.shape}, устройство: {input_batch.device}")
            
            with torch.no_grad():
                outputs = self.model(input_batch)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                top_probs, top_indices = torch.topk(probabilities, top_k)
            
            top_probs = top_probs.cpu().numpy()[0] * 100
            top_indices = top_indices.cpu().numpy()[0]
            
            results = []
            for i in range(top_k):
                breed_name = self.breed_names[top_indices[i]]
                results.append((breed_name, float(top_probs[i])))
            
            print(f"   Результаты: {results}")
            return results
            
        except Exception as e:
            print(f"❌ Ошибка в классификаторе: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            raise
