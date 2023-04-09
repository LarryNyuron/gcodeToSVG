import svgwrite


from scipy.interpolate import CubicSpline
from math import sqrt, isclose


def on_arc(x1, y1, x2, y2, x3, y3):
    '''
    функция, в которой вычисляются координаты центра и радиус дуги на
    основе заданных точек, а затем проверяется, принадлежат ли данные
    точки этой дуге.
    '''

    # вычисляем центр и радиус описанной окружности
    A = x2 - x1
    B = y2 - y1
    C = x3 - x1
    D = y3 - y1
    E = A * (x1 + x2) + B * (y1 + y2)
    F = C * (x1 + x3) + D * (y1 + y3)
    G = 2 * (A * (y3 - y2) - B * (x3 - x2))

    if isclose(G, 0):
        # если G близка к нулю, значит точки лежат на одной прямой
        return False

    # вычисляем координаты центра окружности
    cx = (D * E - B * F) / G
    cy = (A * F - C * E) / G

    # вычисляем радиус окружности
    radius = sqrt((cx - x1) ** 2 + (cy - y1) ** 2)

    # проверяем, принадлежат ли точки дуге
    dist1 = sqrt((x1 - cx) ** 2 + (y1 - cy) ** 2)
    dist2 = sqrt((x2 - cx) ** 2 + (y2 - cy) ** 2)
    dist3 = sqrt((x3 - cx) ** 2 + (y3 - cy) ** 2)

    angle1 = acos((x1 - cx) / radius)
    if y1 - cy < 0:
        angle1 = 2 * pi - angle1

    angle2 = acos((x2 - cx) / radius)
    if y2 - cy < 0:
        angle2 = 2 * pi - angle2

    angle3 = acos((x3 - cx) / radius)
    if y3 - cy < 0:
        angle3 = 2 * pi - angle3

    # проверяем, лежат ли точки на одной дуге
    if isclose(dist1, radius) and isclose(dist2, radius) and isclose(dist3, radius) \
        and angle1 <= angle3 <= angle2:
        return True
    elif isclose(dist1, radius) and isclose(dist2, radius) and isclose(dist3, radius) \
        and angle1 <= angle2 <= angle3:
        return True
    elif isclose(dist1, radius) and isclose(dist3, radius) and isclose(angle1 + angle3, 2 * angle2):
        return True
    elif isclose(dist2, radius) and isclose(dist3, radius) and isclose(angle2 + angle3, 2 * angle1):
        return True
    elif isclose(dist1, radius) and isclose(dist2, radius) and isclose(angle1 + angle2, 2 * angle3):
        return True
    else:
        return False


def rdp(coords, epsilon):
    """
    coords - список координат точек [(x1, y1), (x2, y2), ...]
    epsilon - пороговый параметр

    Реализация алгоритма Рамера-Дугласа-Пекера — это алгоритм, позволяющий
    уменьшить число точек кривой, аппроксимированной большей серией точек.
    Алгоритм был независимо предложен Урсом Рамером в 1972 и Давидом Дугласом
    и Томасом Пекером в 1973. Также алгоритм известен под следующими именами:
    алгоритм Рамера-Дугласа-Пекера, алгоритм итеративной ближайшей точки и
    алгоритм разбиения и слияния. (Из википедии описание)

    сделали, чтобы удаляет из исходного массива точек, которые находятся
    на прямой между точками с крайними координатами, не превышающие заданный
    пороговый параметр.
    """
    if len(coords) < 3:
        return coords

    dmax = 0
    index = 0
    end = len(coords) - 1
    for i in range(1, end):
        d = distance(coords[i], segment(coords[0], coords[end]))
        if d > dmax:
            index = i
            dmax = d

    if dmax > epsilon:
        results_1 = rdp(coords[:index+1], epsilon)
        results_2 = rdp(coords[index:], epsilon)
        results = results_1[:-1] + results_2
    else:
        results = [coords[0], coords[end]]
    return results


def distance(point, line):
    """
    point - координаты точки (x, y)
    line - списки координат двух точек [(x1, y1), (x2, y2)]
    """
    x0, y0 = point
    x1, y1 = line[0]
    x2, y2 = line[1]
    return abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1) / ((y2 - y1)**2 + (x2 - x1)**2)**0.5


def segment(a, b):
    """
    a - первая точка (x1, y1)
    b - вторая точка (x2, y2)
    """
    return [a, b]


def cubic_spline(coords, n_samples):
    """
    coords - список координат точек [(x1, y1), (x2, y2), ...]
    n_samples - количество сэмплов для интерполяции

    Принимает массив координат точек и возвращает упрощенную кривую,
    представленную кубическими сплайнами
    """
    X = [p[0] for p in coords]
    Y = [p[1] for p in coords]

    cs = CubicSpline(X, Y)
    X_interp = [X[0] + (X[-1] - X[0]) * i / (n_samples - 1) for i in range(n_samples)]
    Y_interp = [cs(x) for x in X_interp]

    return [(X_interp[i], Y_interp[i]) for i in range(n_samples)]


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
        print(line)
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
