

def restore_station_name(name: str) -> str:
    current = name
    for _ in range(3):
        fixed = False
        for enc in ('latin-1', 'cp1252'):
            try:
                candidate = current.encode(enc).decode('utf-8')
                if candidate != current:
                    current = candidate
                    fixed = True
                    break
            except (UnicodeEncodeError, UnicodeDecodeError):
                continue
        if not fixed:
            break
    return current


def clean_geothermal_data(input_path: str = 'test-input.csv', output_path: str = 'test-output.csv') -> None:
    records = {}

    with open(input_path, 'r', encoding='utf-8', errors='replace') as infile:
        for line in infile:
            line = line.strip()

            if not line or line.startswith('#'):
                continue

            if line.startswith('reading_id;') or line.startswith('reading_id,'):
                continue

            parts = [p.strip() for p in line.split(';')]
            if len(parts) < 5:
                continue

            try:
                reading_id = int(parts[0])
            except ValueError:
                continue

            raw_station = parts[1]
            station_name = restore_station_name(raw_station)

            raw_temp_str = parts[2].replace(',', '.')
            try:
                raw_temp = float(raw_temp_str)
            except ValueError:
                continue

            unit = parts[3].strip().upper()
            reading_date = parts[4].strip()

            if unit == 'F':
                temp_c = round((raw_temp - 32.0) * 5.0 / 9.0, 1)
            else:
                temp_c = round(raw_temp, 1)

            if abs(temp_c) < 1e-9:
                temp_c = 0.0

            formatted_temp = f'{temp_c:.1f}'

            records[reading_id] = (station_name, formatted_temp, reading_date)

    sorted_ids = sorted(records.keys())

    with open(output_path, 'w', encoding='utf-8', newline='\n') as outfile:
        outfile.write('reading_id,station_name,temperature_c,reading_date\n')
        for rid in sorted_ids:
            st_name, temp_str, date_str = records[rid]
            outfile.write(f'{rid},{st_name},{temp_str},{date_str}\n')


if __name__ == '__main__':
    clean_geothermal_data('test-input.csv', 'test-output.csv')
