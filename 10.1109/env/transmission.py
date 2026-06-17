import math

# --------------------------------------------------
# constants
# --------------------------------------------------
BOLTZMANN = 1.380649e-23
SPEED_OF_LIGHT = 3e8  # سرعت نور برای محاسبه دقیق FSPL


class TransmissionModel:
    """
    پیاده‌سازی کالیبره‌شده مدل انتقال مقاله DVTP
    """

    def __init__(
        self,
        bandwidth=2e6,           # 2 MHz
        frequency=2.4e9,         # 2.4 GHz (Hz)
        transmission_power=30,   # 30 dBm (معادل 1 وات)
        temperature=290,         # 290 Kelvin
    ):
        self.bandwidth = bandwidth
        self.frequency = frequency
        self.transmission_power = transmission_power
        self.temperature = temperature

    def fspl(self, distance_m):
        """
        محاسبه افت مسیر فضای آزاد (FSPL) استاندارد بر حسب متر و هرتز
        """
        distance_m = max(distance_m, 1.0) # جلوگیری از فاصله صفر و خطای لگاریتم
        
        # فرمول استاندارد فیزیکی: 20*log10(4 * pi * d * f / c)
        return 20 * math.log10((4 * math.pi * distance_m * self.frequency) / SPEED_OF_LIGHT)

    def snr(self, distance_m):
        """
        محاسبه دقیق SNR بر حسب dB
        """
        fspl_val = self.fspl(distance_m)

        # توان نویز بر حسب وات: N = k * T * B
        noise_power_watts = BOLTZMANN * self.temperature * self.bandwidth
        # تبدیل توان نویز به dBm: Noise(dBm) = 10 * log10(Noise) + 30
        thermal_noise_dbm = 10 * math.log10(noise_power_watts) + 30

        # SNR (dB) = P_tx (dBm) - FSPL (dB) - Noise (dBm)
        snr_db = self.transmission_power - fspl_val - thermal_noise_dbm
        return snr_db

    def transmission_rate(self, distance_m):
        """
        فرمول شانون برای محاسبه نرخ انتقال (bits/second)
        """
        snr_db = self.snr(distance_m)
        
        # تبدیل SNR از دسی‌بل به حالت خطی
        snr_linear = 10 ** (snr_db / 10)

        # R = B * log2(1 + SNR)
        rate = self.bandwidth * math.log2(1 + snr_linear)

        return max(rate, 1e-3) # حداقل نرخ انتقال بسیار کوچک برای جلوگیری از تقسیم بر صفر

    def transmission_time(self, data_size_kb, distance_m, same_location=False):
        """
        محاسبه زمان انتقال داده (خروجی به ثانیه)
        data_size_kb: اندازه داده به کیلوبایت
        """
        if same_location:
            return 0.0

        # تبدیل کیلوبایت به بیت: KB * 1024 * 8
        data_size_bits = data_size_kb * 1024 * 8
        rate = self.transmission_rate(distance_m)

        return data_size_bits / rate

    def upload_time(self, task_data_kb, distance_m):
        return self.transmission_time(
            data_size_kb=task_data_kb,
            distance_m=distance_m,
            same_location=False
        )

    def download_time(self, result_data_kb, distance_m, is_offloaded=True):
        if not is_offloaded:
            return 0.0
        return self.transmission_time(
            data_size_kb=result_data_kb,
            distance_m=distance_m,
            same_location=False
        )

    @staticmethod
    def euclidean_distance(x1, y1, x2, y2):
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def transmission_threshold_check(self, src_position, dst_position, threshold_sec, data_size_kb):
        distance = self.euclidean_distance(
            src_position[0], src_position[1],
            dst_position[0], dst_position[1]
        )
        t = self.transmission_time(data_size_kb=data_size_kb, distance_m=distance)
        return t <= threshold_sec


if __name__ == "__main__":
    model = TransmissionModel()

    print("=" * 60)
    print("Modified Transmission Rate Test (Realistic Output)")
    print("=" * 60)

    for d in [1, 10, 25, 50, 75, 100, 150, 200, 500]:
        fspl_v = model.fspl(d)
        snr_v = model.snr(d)
        rate_v = model.transmission_rate(d) / 1e6  # تبدیل به Mbps

        print(
            f"Distance={d:>4} m | "
            f"FSPL={fspl_v:>7.2f} dB | "
            f"SNR={snr_v:>7.2f} dB | "
            f"Rate={rate_v:>8.2f} Mbps"
        )