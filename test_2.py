# Определение дуги/окружности и преобразование ее в элемент path svg
# Этот код может выполняться в рамках класса GCodeParser, отвечающего за парсинг исходной программы
def convert_arc_to_path(self, x, y, i, j, clockwise):

    start_angle = math.atan2(-j, -i)
    end_angle = math.atan2(y - j, x - i)
    radius = math.sqrt((x - i) ** 2 + (y - j) ** 2)

    if clockwise:
        start_angle, end_angle = end_angle, start_angle

    sweep_flag = abs(end_angle - start_angle) <= math.pi
    large_arc_flag = abs(end_angle - start_angle) >= math.pi

    start_x, start_y = self.current_position
    end_x, end_y = x, y

    path = f"A {radius}, {radius} 0 {int(large_arc_flag)}, {int(sweep_flag)} {end_x}, {end_y}"
    self.current_position = (end_x, end_y)

    return path

# Аппроксимация сплайнами
# Этот код может выполняться в рамках класса PathApproximator, отвечающего за аппроксимацию пути
def approximate_path(self, path, spline_type, accuracy):

    # Разбиваем путь на отдельные сегменты
    segments = self.split_path_into_segments(path)

    # Проходим по каждому сегменту и аппроксимируем его сплайном
    approximated_path = []
    for segment in segments:
        if segment[0] == "L" or segment[0] == "M":
            # Это обычный отрезок - добавляем его без изменений
            approximated_path.append(segment)
        elif segment[0] == "A":
            # Это дуга - конвертируем ее в сплайн
            x1, y1, radius_x, radius_y, angle, large_arc_flag, sweep_flag, x2, y2 = self.parse_arc_segment(segment)
            approximated_segment = self.approximate_arc(x1, y1, radius_x, radius_y, angle, large_arc_flag, sweep_flag, x2, y2, spline_type, accuracy)
            approximated_path.extend(approximated_segment)
    return " ".join(approximated_path)

# Прямое упрощение геометрии
# Этот код может выполняться в рамках класса GeometrySimplifier, отвечающего за упрощение геометрии
def simplify_geometry(self, path, max_distance):

    # Создаем объекты точек и группируем их в контуры
    points = self.parse_path_into_points(path)
    contours = self.group_points_into_contours(points)

    # Проходим по каждому контуру и упрощаем его
    simplified_path = []
    for contour in contours:
        # Соединяем точки, находящиеся на расстоянии меньше max_distance
        new_contour_points = self.connect_points_within_distance(contour, max_distance)
        # Если контур был закрыт (последняя точка равна первой), то добавляем замыкающую линию
        if new_contour_points[0] == new_contour_points[-1]:
            new_contour_points.append(new_contour_points[0])
        # Конвертируем точки обратно в путь
        new_contour_path = self.convert_points_to_path(new_contour_points)
        simplified_path.append(new_contour_path)
    return " ".join(simplified_path)