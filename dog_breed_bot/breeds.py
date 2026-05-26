# breeds.py
def load_breed_names(file_path="breed_names.txt"):
    """Загружает список пород из файла и очищает названия."""
    breeds = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_breeds = [line.strip() for line in f.readlines() if line.strip()]
        
        # Очищаем названия пород: оставляем только имя после дефиса
        for breed in raw_breeds:
            if '-' in breed:
                clean_name = breed.split('-')[-1].replace('_', ' ')
                breeds.append(clean_name)
            else:
                breeds.append(breed)
        
        print(f"✅ Загружено {len(breeds)} пород из {file_path}")
        if breeds:
            print(f"   Пример: {breeds[0]}, {breeds[1]}, {breeds[2]}")
    except FileNotFoundError:
        print(f"❌ Файл {file_path} не найден!")
        breeds = [f"Breed_{i}" for i in range(120)]
    return breeds