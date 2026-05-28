import os
import random
import shutil

def random_rename_images(folder_path):
    # Поддерживаемые расширения изображений
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp')
    
    # Получаем список всех файлов изображений в папке
    files = [f for f in os.listdir(folder_path)
             if os.path.isfile(os.path.join(folder_path, f)) and
             f.lower().endswith(image_extensions)]
    
    if not files:
        print("❌ В папке не найдено изображений.")
        return
    
    print(f"📁 Найдено изображений: {len(files)}")
    
    # Перемешиваем список файлов
    random.shuffle(files)
    
    # Временная папка для избежания конфликтов при переименовании
    temp_folder = os.path.join(folder_path, '_temp_rename')
    os.makedirs(temp_folder, exist_ok=True)
    
    # Шаг 1: переименовываем во временные имена
    temp_names = []
    for i, filename in enumerate(files, start=1):
        old_path = os.path.join(folder_path, filename)
        temp_name = f'_temp_{i:04d}{os.path.splitext(filename)[1]}'
        temp_path = os.path.join(temp_folder, temp_name)
        shutil.move(old_path, temp_path)
        temp_names.append((temp_path, os.path.splitext(filename)[1]))
        print(f"➡️ Временно: {filename} → {temp_name}")
    
    # Шаг 2: переименовываем в финальные имена 0001.jpg и т.д.
    for i, (temp_path, ext) in enumerate(temp_names, start=1):
        new_name = f'{i:04d}.jpg'  #统一扩展名为 .jpg
        new_path = os.path.join(folder_path, new_name)
        shutil.move(temp_path, new_path)
        print(f"✅ Финально: → {new_name}")
    
    # Удаляем временную папку
    os.rmdir(temp_folder)
    
    print(f"\n🎉 Готово! {len(files)} изображений переименовано в случайном порядке.")

if __name__ == '__main__':
    # Укажите путь к папке с изображениями
    folder = input("Введите путь к папке с фотографиями: ").strip()
    
    if os.path.isdir(folder):
        random_rename_images(folder)
    else:
        print("❌ Указанная папка не существует.")