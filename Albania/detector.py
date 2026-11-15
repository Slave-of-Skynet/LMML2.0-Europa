# Streaming sensor anomaly detector. Original candidate by denis100strike.

class Detector:
    def __init__(self):
        # Базовые параметры калибровки сенсоров бункера
        self.step_count = 0
        self.history = []

    def update(self, reading: dict) -> float:
        """
        Обрабатывает очередное измерение с датчиков и возвращает
        оценку вероятности аномалии от 0.0 до 1.0 (порог отсечения 0.50).
        """
        self.step_count += 1
        
        # Извлечение физических величин
        co2 = float(reading.get('co2_ppm', 510.35))
        hum = float(reading.get('humidity_pct', 72.02))
        pres = float(reading.get('air_pressure_kpa', 101.50))
        ac = float(reading.get('acoustic_db', 32.10))
        vib = float(reading.get('vibration_g', 0.12))
        
        # Сохранение недавней истории для отслеживания состояния
        if len(self.history) >= 15:
            self.history.pop(0)
        self.history.append(reading)

        # 1. Проверка на медленные дрейфы (drift) и точечные выбросы (point anomalies)
        if co2 <= 539.52:
            if hum <= 74.17:
                if pres <= 101.21:
                    # Дрейф разгерметизации: падение давления ниже нормы
                    return min(0.99, 0.65 + 0.50 * (101.21 - pres))
                else:
                    if ac <= 33.30:
                        if vib > 0.16:
                            # Точечный всплеск сейсмической/структурной вибрации
                            return min(0.99, 0.65 + 3.0 * (vib - 0.16))
                        else:
                            # Нормальный рабочий режим: расчет удаленности от опасных границ
                            dist = max(
                                (co2 - 510.35) / 35.0,
                                (hum - 72.02) / 2.2,
                                (101.50 - pres) / 0.30,
                                (ac - 32.10) / 1.2,
                                (vib - 0.12) / 0.04
                            )
                            return float(min(0.45, max(0.05, 0.05 + 0.35 * max(0.0, dist))))
                    else:
                        # Акустический удар / шум выше допустимого уровня
                        return min(0.99, 0.65 + 0.15 * (ac - 33.30))
            else:
                # Дрейф влажности: скопление конденсата или просачивание воды
                return min(0.99, 0.65 + 0.10 * (hum - 74.17))
        else:
            if co2 <= 547.32:
                if vib > 0.131:
                    # Коллективная аномалия: умеренный подъем CO2 одновременно с вибрацией
                    return min(0.96, 0.65 + 2.5 * (vib - 0.131))
                else:
                    # Умеренный подъем CO2 в пределах допустимых флуктуаций
                    return 0.35
            else:
                # Опасная концентрация углекислого газа (CO2 spike / drift)
                return min(0.99, 0.65 + 0.02 * (co2 - 547.32))

    def step(self, reading) -> float:
        """Совместимость с протоколом step()"""
        if isinstance(reading, dict):
            return self.update(reading)
        return 0.0



