import time


class Timer:
    def __init__(self, batch_size=1):
        assert batch_size > 0
        self.batch_size = batch_size
        self.total_time = 0
        self.total_sample = 0
        self.last_start_time = None
        self.last_round_time = 0  # used to calculate only the last epoch/round time used

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def start_timer(self):
        assert self.last_start_time is None
        self.last_start_time = time.time()

    def stop_timer(self):
        assert self.last_start_time is not None
        self.last_round_time = time.time() - self.last_start_time
        self.total_time += self.last_round_time
        self.total_sample += self.batch_size
        self.last_start_time = None

    def clear_timer(self):
        self.total_time = 0
        self.total_sample = 0
        self.last_start_time = None
        self.last_round_time = 0

    def get_average_time(self):
        if self.total_time == 0:
            return 0
        return self.total_time / self.total_sample

    def get_average_speed(self):
        if self.total_sample == 0:
            return 0
        return self.total_sample / self.total_time

    def get_last_round_time(self):
        return self.last_round_time / self.batch_size

    def get_last_round_speed(self):
        if self.last_round_time == 0:
            return 0
        return self.batch_size / self.last_round_time

    def get_string(self):
        return "Timer ==> Time Used: {}s\t\tSpeed: {} ({}) sample/s".format(
            self.total_time,
            self.get_average_speed(),
            self.get_last_round_speed(),
        )


global_timer = Timer()

def get_global_timer():
    return global_timer

def set_global_timer_batch_size(batch_size):
    global_timer.set_batch_size(batch_size)