import math


# --------------------------------------------------
# constants
# --------------------------------------------------

BOLTZMANN = 1.380649e-23


class TransmissionModel:
    """
    Implements equations (1) - (9) from the paper.

    Includes:
    - FSPL
    - SNR
    - transmission rate
    - transmission time
    - upload time
    - download time
    """

    def __init__(
        self,
        bandwidth=2e6,          # 2 MHz
        frequency=2.4e9,        # 2.4 GHz
        transmission_power=30,  # dBm
        temperature=290,        # Kelvin
        alpha=32.44             # FSPL constant
    ):

        self.bandwidth = bandwidth
        self.frequency = frequency
        self.transmission_power = transmission_power
        self.temperature = temperature
        self.alpha = alpha

    # ==================================================
    # equation (1)
    # ==================================================

    def fspl(self, distance_m):

        distance_m = max(distance_m, 1.0)
        distance_km = distance_m / 1000.0
        frequency_mhz = self.frequency / 1e6

        return (
            20 * math.log10(distance_km)
            + 20 * math.log10(frequency_mhz)
            + self.alpha
        )

    # ==================================================
    # equation (2)
    # ==================================================

    def snr(self, distance_m):

        fspl = self.fspl(distance_m)

        thermal_noise_dbm = (
            10
            *
            math.log10(
                BOLTZMANN
                *
                self.temperature
                *
                self.bandwidth
            )
            +
            30
        )

        return (
            self.transmission_power
            - fspl
            - thermal_noise_dbm
        )

    # ==================================================
    # equation (3)
    # ==================================================

    def transmission_rate(
        self,
        distance_m
    ):
        """
        Shannon capacity
        """

        snr_db = self.snr(distance_m)

        snr_linear = 10 ** (snr_db / 10)

        rate = (
            self.bandwidth
            * math.log2(1 + snr_linear)
        )

        return max(rate, 1e-6)

    # ==================================================
    # equation (4)
    # ==================================================

    def transmission_time(
        self,
        data_size_kb,
        distance_m,
        same_location=False
    ):
        """
        data_size_kb : KB
        return : seconds
        """

        if same_location:
            return 0.0

        data_size_bits = data_size_kb * 1024 * 8

        rate = self.transmission_rate(distance_m)

        return data_size_bits / rate

    # ==================================================
    # equation (5)
    # ==================================================

    def upload_time(
        self,
        task_data_kb,
        distance_m
    ):

        return self.transmission_time(
            data_size_kb=task_data_kb,
            distance_m=distance_m,
            same_location=False
        )

    # ==================================================
    # equation (8)
    # ==================================================

    def download_time(
        self,
        result_data_kb,
        distance_m,
        is_offloaded=True
    ):

        if not is_offloaded:
            return 0.0

        return self.transmission_time(
            data_size_kb=result_data_kb,
            distance_m=distance_m,
            same_location=False
        )

    # ==================================================
    # equation (9)
    # ==================================================

    @staticmethod
    def euclidean_distance(
        x1,
        y1,
        x2,
        y2
    ):

        return math.sqrt(
            (x1 - x2) ** 2 +
            (y1 - y2) ** 2
        )

    def transmission_threshold_check(
        self,
        src_position,
        dst_position,
        threshold_sec,
        data_size_kb
    ):

        distance = self.euclidean_distance(
            src_position[0],
            src_position[1],
            dst_position[0],
            dst_position[1]
        )

        t = self.transmission_time(
            data_size_kb=data_size_kb,
            distance_m=distance
        )

        return t <= threshold_sec


if __name__ == "__main__":

    model = TransmissionModel()

    distance = 100

    rate = model.transmission_rate(distance)

    print(f"Transmission Rate: {rate / 1e6:.2f} Mbps")

    tx_time = model.transmission_time(
        data_size_kb=500,
        distance_m=distance
    )

    print(f"Transmission Time: {tx_time:.4f} sec")
