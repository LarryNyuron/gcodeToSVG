import argparse
import numpy as np
import svgwrite
from scipy.interpolate import splprep, splev

def parse_command_line():
    parser = argparse.ArgumentParser(description='Convert G-code to SVG-like format')
    parser.add_argument('input_file', metavar='input', type=str, help='Input G-code file')
    parser.add_argument('output_file', metavar='output', type=str, help='Output SVG file')
    parser.add_argument('--spline', type=str, default='cubic', help='Type of spline (cubic or quadratic)')
    parser.add_argument('--tolerance', type=float, default=0.1, help='Tolerance for spline approximation')
    return parser.parse_args()

def read_gcode(filename):
    coords = []
    with open(filename, 'r') as f:
        for line in f:
            parts = line.split()
            if len(parts) > 1 and parts[0] in ['G0', 'G1']:
                coord = [float(p[1:]) for p in parts[1:] if p[0] in 'XY']
                if coord:
                    coords.append(coord)
    return np.array(coords)

def to_svg(coords, spline_type='cubic', tolerance=0.1):
    dwg = svgwrite.Drawing('output.svg', profile='full')
    for i in range(len(coords) - 1):
        x1, y1 = coords[i]
        x2, y2 = coords[i + 1]
        path = dwg.add(dwg.path(stroke=svgwrite.rgb(0, 0, 0, '%')))
        if spline_type == 'cubic':
            # Аппроксимация кубическим сплайном
            path.push('M', x1, y1, 'C')
            xc, yc = splprep([coords[:, 0], coords[:, 1]], s=tolerance)[0]
            x, y = splev(np.linspace(0, 1, 100), (xc, yc))
            for j in range(1, len(x)):
                path.push(x[j], y[j])
        elif spline_type == 'quadratic':
            # Аппроксимация квадратичным сплайном
            path.push('M', x1, y1, 'Q')
            xc, yc = splprep([coords[:, 0], coords[:, 1]], k=2, s=tolerance)[0]
            x, y = splev(np.linspace(0, 1, 100), (xc, yc))
            for j in range(1, len(x)):
                path.push(x[j], y[j])
        path.push('L', x2, y2)
    dwg.save()

if __name__ == '__main__':
    args = parse_command_line()
    coords = read_gcode(args.input_file)
    to_svg(coords, args.spline, args.tolerance)