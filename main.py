import svgwrite

# Открыть файл с G-кодом для чтения
with open('input.gcode', 'r') as f:
    # Создание нового документа SVG
    dwg = svgwrite.Drawing('output.svg', profile='full')

    # Начальные координаты
    x = 0
    y = 0

    # Цикл по строкам в G-коде
    for line in f:
        # Получение координат следующей точки
        if line.startswith('G0'):
            # Если команда G0, то перемещение без позиционирования,
            # поэтому берем только координаты следующей точки
            parts = line.split()
            x, y = float(parts[1][1:]), float(parts[2][1:])
        elif line.startswith('G1'):
            # Если команда G1, то линия в заданные координаты,
            # поэтому берем координаты текущей и следующей точки
            parts = line.split()
            x1, y1 = x, y
            x2, y2 = float(parts[1][1:]), float(parts[2][1:])
            # Создание линии в SVG
            dwg.add(dwg.line((x1, y1), (x2, y2), stroke=svgwrite.rgb(0, 0, 0, '%')))
            # Обновление текущих координат
            x, y = x2, y2

    # Сохранение SVG-документа
    dwg.save()
