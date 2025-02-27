model = YOLO("yolov8n.pt")

def load_model(model_path="yolov8n.pt"):
    """
    Загружает предобученную модель YOLOv8.

    Args:
        model_path (str): Путь к файлу модели YOLOv8.

    Returns:
        YOLO: Загруженная модель YOLOv8.
    """
    # Проверяем существование файла модели
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Модель {model_path} не найдена!")
    return YOLO(model_path)

def process_video(input_path, output_path, model):
    """
    Обрабатывает видео, выполняя детекцию людей и сохраняя результат.

    Args:
        input_path (str): Путь к входному видеофайлу.
        output_path (str): Путь для сохранения обработанного видео.
        model (YOLO): Модель YOLOv8 для детекции.

    Returns:
        tuple: Количество кадров.
    """
    # Открываем исходное видео
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Не удалось открыть видео {input_path}")

    # Получаем параметры видео
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Настраиваем запись выходного видео
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    frame_count = 0
    person_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Выполняем детекцию
        results = model(frame, classes=[0], verbose=False)  # Класс 0 - это "person" в COCO

        # Отрисовываем результаты
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0]
                label = f"Person: {conf:.2f}"

                # Увеличиваем толщину линии и размер текста
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                           0.6, (0, 255, 0), 2)

        # Записываем обработанный кадр
        out.write(frame)

        if frame_count % 50 == 0:
          print(f"Обработано кадров: {frame_count}")

    # Освобождаем ресурсы
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    return frame_count

def main():
    """Точка входа в программу."""
    input_path = "crowd.mp4"
    output_path = "output/processed_crowd.mp4"
    model_path = "yolov8n.pt"

    # Создаем выходную директорию, если ее нет
    os.makedirs("output", exist_ok=True)

    try:
        # Загружаем модель
        model = load_model(model_path)

        # Обрабатываем видео
        frames = process_video(input_path, output_path, model)

        # Выводим статистику
        print(f"\nОбработка завершена!")
        print(f"Обработано кадров: {frames}")

    except Exception as e:
        print(f"Ошибка: {str(e)}")

if __name__ == "__main__":
    main()
