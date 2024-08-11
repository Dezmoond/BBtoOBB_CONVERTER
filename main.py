from ultralytics import SAM
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
# from IPython.display import display, Image
import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageChops
import matplotlib.patches as patches
import os
import threading
from matplotlib.widgets import Slider  # test delet
from ultralytics.models.sam import Predictor as SAMPredictor
from matplotlib.widgets import Button
import matplotlib
matplotlib.use('TkAgg')
# Словарь для цветов классов
CLASS_COLORS = {
    0: 'red',
    1: 'blue',
    2: 'green',
    3: 'orange',
    4: 'purple',
    5: 'brown',
    6: 'pink',
    7: 'gray',
    8: 'cyan',
    9: 'magenta',
}


def load_model(model_path):
    return SAM(model_path)


def read_annotations_from_file(file_path):
    annotations = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.strip():
                parts = line.split()
                print(f"Processing line: {line.strip()}")
                if len(parts) == 5:
                    try:
                        annotations.append([float(part) for part in parts])
                    except ValueError:
                        print(f"Skipping invalid line: {line.strip()}")
                else:
                    print(f"Invalid format: {line.strip()}")
    return annotations


def read_annotations_from_file_OBB(file_path):
    annotations_obb = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.strip():
                parts = line.split()
                print(f"Processing line: {line.strip()}")
                if len(parts) == 9:
                    try:
                        annotations_obb.append([float(part) for part in parts])
                        print('OBB')
                        type_ann = 1
                    except ValueError:
                        print(f"Skipping invalid line: {line.strip()}")
                else:
                    if len(parts) == 5:
                        try:
                            annotations_obb.append([float(part) for part in parts])
                            type_ann = 0
                        except ValueError:
                            print(f"Skipping invalid line: {line.strip()}")
                    else:
                        print(f"Invalid format: {line.strip()}")
    return annotations_obb, type_ann


def denormalize_and_convert(cx, cy, rw, rh, img_width, img_height):
    x_center = cx * img_width
    y_center = cy * img_height
    box_width = rw * img_width
    box_height = rh * img_height

    x_min = int(x_center - box_width / 2)
    y_min = int(y_center - box_height / 2)
    x_max = int(x_center + box_width / 2)
    y_max = int(y_center + box_height / 2)

    return [x_min, y_min, x_max, y_max]


def get_mask_outline(mask_array):
    kernel = np.ones((5, 5), np.uint8)
    dilated_mask = cv2.dilate(mask_array, kernel, iterations=1)
    outline = dilated_mask - mask_array
    outline_color = 255
    highlighted_mask = np.zeros_like(mask_array)
    highlighted_mask[mask_array > 0] = outline_color
    fatmask = cv2.add(highlighted_mask, outline)
    return fatmask


def get_bbox_from_mask(fatmask):
    contours, _ = cv2.findContours(fatmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        print("Контуры не найдены.")
        return None

    all_points = np.concatenate(contours)
    rect = cv2.minAreaRect(all_points)
    box = cv2.boxPoints(rect)
    box = np.intp(box)
    return box


def normalize_coordinates(coords, img_width, img_height):
    return [(x / img_width, y / img_height) for x, y in zip(coords[0::2], coords[1::2])]


def process_annotations(model, image_path, annotations, img_width, img_height):
    final_annotations = []

    for a in annotations:
        clas, cx, cy, rw, rh = a
        clas = int(clas)
        print(clas)

        bbox_pixel = denormalize_and_convert(cx, cy, rw, rh, img_width, img_height)
        results = model(image_path, bboxes=[bbox_pixel])

        mask = results[0].masks.data[0]
        mask_array = mask.cpu().numpy().astype(np.uint8) * 255
        fatmask = get_mask_outline(mask_array)
        box = get_bbox_from_mask(fatmask)

        if box is None:
            print("Контур не найден, используем BB для OBB.")
            x1, y1 = bbox_pixel[0], bbox_pixel[1]
            x2, y2 = bbox_pixel[0] + bbox_pixel[2], bbox_pixel[1]
            x3, y3 = bbox_pixel[0] + bbox_pixel[2], bbox_pixel[1] + bbox_pixel[3]
            x4, y4 = bbox_pixel[0], bbox_pixel[1] + bbox_pixel[3]
        else:
            x1, y1 = box[0]
            x2, y2 = box[1]
            x3, y3 = box[2]
            x4, y4 = box[3]

        coordinates = f"{x1} {y1} {x2} {y2} {x3} {y3} {x4} {y4}"
        print("Координаты углов прямоугольника:", coordinates)

        annotations_list = [float(x) for x in coordinates.split()]
        normalized_coords = normalize_coordinates(annotations_list, img_width, img_height)

        flattened_coords = [coord for pair in normalized_coords for coord in pair]
        final_list = [clas] + flattened_coords
        final_annotations.append(final_list)

    return final_annotations


def save_annotations_to_file(annotations, output_path):
    with open(output_path, 'w') as file:
        for annotation in annotations:
            annotation_str = ' '.join(map(str, annotation))
            file.write(annotation_str + '\n')


def draw_bbox_on_image(image, annotations, img_height, img_width):
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    for annotation in annotations:
        _, x1, y1, x2, y2, x3, y3, x4, y4 = annotation
        coords = [x1, y1, x2, y2, x3, y3, x4, y4]
        print(f"Normalized coordinates: {coords}")

        denormalized_coords = [
            (x * img_width, y * img_height)
            for x, y in zip(coords[0::2], coords[1::2])
        ]
        print(f"Denormalized coordinates: {denormalized_coords}")

        polygon = patches.Polygon(denormalized_coords, closed=True, edgecolor='green', linewidth=3, fill=False)
        ax.add_patch(polygon)

    plt.show()




class IndexTracker:
    def __init__(self, ax, image_paths, annotations_list, img_heights, img_widths):
        self.ax = ax
        self.image_paths = image_paths
        self.annotations_list = annotations_list
        self.img_heights = img_heights
        self.img_widths = img_widths
        self.idx = 0
        self.fig = ax.figure
        self.update_plot()

        # Кнопки для переключения изображений
        axprev = plt.axes([0.7, 0.03, 0.05, 0.055])
        axnext = plt.axes([0.81, 0.03, 0.05, 0.055])
        self.bnext = Button(axnext, 'Вперед')
        self.bprev = Button(axprev, 'Назад')
        self.bnext.on_clicked(self.next)
        self.bprev.on_clicked(self.prev)

    def update_plot(self):
        self.ax.clear()
        img = cv2.cvtColor(cv2.imread(self.image_paths[self.idx]), cv2.COLOR_BGR2RGB)
        self.ax.imshow(img)

        annotations, type_ann = self.annotations_list[self.idx]
        self.draw_annotations_obb(annotations, self.img_widths[self.idx], self.img_heights[self.idx], type_ann)

    def draw_annotations_obb(self, annotations, img_width, img_height, type_ann):
        for annotation in annotations:
            clas = int(annotation[0])
            coords = annotation[1:]
            if type_ann == 0:  # BB
                cx, cy, rw, rh = coords
                x_center = cx * img_width
                y_center = cy * img_height
                box_width = rw * img_width
                box_height = rh * img_height
                x_min = int(x_center - box_width / 2)
                y_min = int(y_center - box_height / 2)
                x_max = int(x_center + box_width / 2)
                y_max = int(y_center + box_height / 2)
                denormalized_coords = [x_min, y_min, x_max, y_min, x_max, y_max, x_min, y_max]
                polygon = patches.Polygon(
                    [denormalized_coords[i:i + 2] for i in range(0, len(denormalized_coords), 2)],
                    closed=True, edgecolor=CLASS_COLORS.get(clas, 'black'), linewidth=3, fill=False)
                self.ax.add_patch(polygon)
                self.ax.text(x_min, y_min, f'Class {clas}', color=CLASS_COLORS.get(clas, 'black'),
                             fontsize=12, verticalalignment='top',
                             bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))
            elif type_ann == 1:  # OBB
                x1, y1, x2, y2, x3, y3, x4, y4 = coords
                denormalized_coords = [
                    (x * img_width, y * img_height)
                    for x, y in zip([x1, x2, x3, x4], [y1, y2, y3, y4])
                ]
                polygon = patches.Polygon(denormalized_coords, closed=True,
                                          edgecolor=CLASS_COLORS.get(clas, 'black'), linewidth=3, fill=False)
                self.ax.add_patch(polygon)
                self.ax.text(denormalized_coords[0][0], denormalized_coords[0][1], f'Class {clas}',
                            color=CLASS_COLORS.get(clas, 'black'), fontsize=12, verticalalignment='top',
                            bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))

        self.ax.set_title(f"File: {os.path.basename(self.image_paths[self.idx])} | Type: {'OBB' if type_ann == 1 else 'BB'}")
        self.fig.canvas.draw_idle()

    def next(self, event):
        self.idx = (self.idx + 1) % len(self.image_paths)
        self.update_plot()

    def prev(self, event):
        self.idx = (self.idx - 1) % len(self.image_paths)
        self.update_plot()

# Функция для отображения изображений с аннотациями OBB
def display_images_with_annotations_OBB(image_paths, annotations_list, img_heights, img_widths):
    plt.close('all')  # Закрываем все предыдущие окна
    fig, ax = plt.subplots(figsize=(10, 10))
    fig.canvas.manager.set_window_title('Просмотр')
    tracker = IndexTracker(ax, image_paths, annotations_list, img_heights, img_widths)
    plt.show()
def load_model_with_progress(model_path, progress_bar, status_label):
    status_label.config(text="Загрузка модели...")
    progress_bar.start(10)  # Начинаем анимацию прогресс-бара
    model = SAM(model_path)
    progress_bar.stop()  # Останавливаем анимацию прогресс-бара
    status_label.config(text="Модель загружена.")
    return model


def main(image_folder, annotation_folder, progress_bar, status_label):
    model_path = "sam2_b.pt"

    # Создание выходной папки на основе имени папки с аннотациями
    annotation_folder_name = os.path.basename(annotation_folder.rstrip("/"))
    output_folder = os.path.join(os.path.dirname(annotation_folder), annotation_folder_name + '_OBB')
    os.makedirs(output_folder, exist_ok=True)

    # Загрузка модели с прогресс-баром
    model = load_model_with_progress(model_path, progress_bar, status_label)

    image_paths = []
    annotations_list = []
    img_heights = []
    img_widths = []

    for file_name in os.listdir(image_folder):
        if file_name.endswith('.jpg'):
            image_path = os.path.join(image_folder, file_name)
            annotation_file_name = file_name.replace('.jpg', '.txt')
            annotations_path = os.path.join(annotation_folder, annotation_file_name)

            if not os.path.exists(annotations_path):
                print(f"Файл аннотаций {annotations_path} не существует, пропускаем.")
                continue

            annotations = read_annotations_from_file(annotations_path)
            print(f"Аннотации для {file_name}: {annotations}")

            img = cv2.imread(image_path)
            img_height, img_width = img.shape[:2]

            final_annotations = process_annotations(model, image_path, annotations, img_width, img_height)
            print(f'Список OBB для {file_name}:', final_annotations)

            output_annotations_path = os.path.join(output_folder, annotation_file_name)
            save_annotations_to_file(final_annotations, output_annotations_path)
            print(f'Нормализованные аннотации для {file_name} сохранены в файл: {output_annotations_path}')

            image_paths.append(image_path)
            annotations_list.append(final_annotations)
            img_heights.append(img_height)
            img_widths.append(img_width)

    status_label.config(text="Завершено.")



def select_folder(entry):
    folder_selected = filedialog.askdirectory()
    entry.delete(0, tk.END)
    entry.insert(0, folder_selected)


def process_images_and_annotations(model, image_folder, annotation_folder, progress_bar, status_label):
    # Создание выходной папки на основе имени папки с аннотациями
    annotation_folder_name = os.path.basename(annotation_folder.rstrip("/"))
    output_folder = os.path.join(os.path.dirname(annotation_folder), annotation_folder_name + '_OBB')
    os.makedirs(output_folder, exist_ok=True)

    image_paths = []
    annotations_list = []
    img_heights = []
    img_widths = []

    # Получаем список всех изображений для оценки количества итераций
    images = [file_name for file_name in os.listdir(image_folder) if file_name.endswith('.jpg')]
    total_images = len(images)
    progress_bar['maximum'] = total_images

    for idx, file_name in enumerate(images):
        image_path = os.path.join(image_folder, file_name)
        annotation_file_name = file_name.replace('.jpg', '.txt')
        annotations_path = os.path.join(annotation_folder, annotation_file_name)

        if not os.path.exists(annotations_path):
            print(f"Файл аннотаций {annotations_path} не существует, пропускаем.")
            continue

        annotations = read_annotations_from_file(annotations_path)
        print(f"Аннотации для {file_name}: {annotations}")

        img = cv2.imread(image_path)
        img_height, img_width = img.shape[:2]

        final_annotations = process_annotations(model, image_path, annotations, img_width, img_height)
        print(f'Список OBB для {file_name}:', final_annotations)

        output_annotations_path = os.path.join(output_folder, annotation_file_name)
        save_annotations_to_file(final_annotations, output_annotations_path)
        print(f'Нормализованные аннотации для {file_name} сохранены в файл: {output_annotations_path}')

        image_paths.append(image_path)
        annotations_list.append(final_annotations)
        img_heights.append(img_height)
        img_widths.append(img_width)

        # Обновление прогресс-бара и статуса
        progress_bar.step(1)
        status_label.config(text=f"Обработка {idx + 1}/{total_images}...")

    status_label.config(text="Обработка завершена.")
    preview_obb(image_folder, f'{annotation_folder}_OBB')


def run_script(image_folder_entry, annotation_folder_entry, progress_bar, status_label):
    image_folder = image_folder_entry.get()
    annotation_folder = annotation_folder_entry.get()

    if not image_folder or not annotation_folder:
        messagebox.showwarning("Требуется ввод", "Пожалуйста, выберите обе папки перед запуском скрипта.")
        return

    def thread_target():
        status_label.config(text="Запуск...")
        model = load_model_with_progress("sam2_b.pt", progress_bar, status_label)
        process_images_and_annotations(model, image_folder, annotation_folder, progress_bar, status_label)

    # Создание и запуск потока для выполнения основной задачи
    processing_thread = threading.Thread(target=thread_target)
    processing_thread.start()


def preview_obb(image_folder, annotation_folder):
    image_paths = []
    annotations_list = []
    img_heights = []
    img_widths = []
    for file_name in os.listdir(image_folder):
        if file_name.endswith('.jpg'):
            image_path = os.path.join(image_folder, file_name)
            annotation_file_name = file_name.replace('.jpg', '.txt')
            annotations_path = os.path.join(annotation_folder, annotation_file_name)

            if not os.path.exists(annotations_path):
                print(f"Файл аннотаций {annotations_path} не существует, пропускаем.")
                continue

            annotations = read_annotations_from_file_OBB(annotations_path)
            print(f"Аннотации для {file_name}: {annotations}")

            img = cv2.imread(image_path)
            img_height, img_width = img.shape[:2]

            image_paths.append(image_path)
            annotations_list.append(annotations)
            img_heights.append(img_height)
            img_widths.append(img_width)
    print(image_paths)
    print(annotations_list)
    display_images_with_annotations_OBB(image_paths, annotations_list, img_heights, img_widths)




def create_gui():
    root = tk.Tk()
    root.title("BB to OBB")

    # Выбор папки с изображениями
    tk.Label(root, text="Выберите папку с изображениями:").grid(row=0, column=0, padx=10, pady=10)
    image_folder_entry = tk.Entry(root, width=50)
    image_folder_entry.grid(row=0, column=1, padx=10, pady=10)
    tk.Button(root, text="Обзор", command=lambda: select_folder(image_folder_entry)).grid(row=0, column=2, padx=10,
                                                                                          pady=10)

    # Выбор папки с аннотациями
    tk.Label(root, text="Выберите папку с аннотациями:").grid(row=1, column=0, padx=10, pady=10)
    annotation_folder_entry = tk.Entry(root, width=50)
    annotation_folder_entry.grid(row=1, column=1, padx=10, pady=10)
    tk.Button(root, text="Обзор", command=lambda: select_folder(annotation_folder_entry)).grid(row=1, column=2, padx=10,
                                                                                               pady=10)

    # Кнопка запуска
    run_button = tk.Button(root, text="Запуск конвертации",
                           command=lambda: run_script(image_folder_entry, annotation_folder_entry, progress_bar,
                                                      status_label))
    run_button.grid(row=2, columnspan=3, pady=10)
    # Кнопка просмотра OBB
    preview_button = tk.Button(root, text="Просмотр BB и OBB",
                               command=lambda: preview_obb(image_folder_entry.get(), annotation_folder_entry.get()))
    preview_button.grid(row=3, columnspan=3, pady=10)

    # Прогресс-бар
    progress_bar = ttk.Progressbar(root, orient="horizontal", mode="indeterminate", length=400)
    progress_bar.grid(row=4, columnspan=3, pady=10)

    # Метка статуса
    status_label = tk.Label(root, text="Готово к работе.")
    status_label.grid(row=5, columnspan=3, pady=10)

    root.mainloop()


if __name__ == "__main__":

    create_gui()
