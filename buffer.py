import numpy as np

class SampleBuffer():
    def __init__(self, size, obs_size, act_size, gamma):
        self.gamma = gamma
        self.size = size
        self.s = np.zeros(shape=(size, obs_size), dtype=np.float32)
        self.s_ = np.zeros(shape=(size, obs_size), dtype=np.float32)
        self.a = np.zeros(shape=(size, act_size), dtype=np.float32)
        self.a_lp = np.zeros(shape=(size, act_size), dtype=np.float32)
        self.r = np.zeros(shape=(size), dtype=np.float32)
        self.ret = np.zeros(shape=(size), dtype=np.float32)
        self.end_ptr = 0
        self.start_ptr = 0

    def clear(self):
        self.s[:] = 0
        self.a[:] = 0
        self.a_lp[:] = 0
        self.r[:] = 0
        self.s[:] = 0
        self.ret[:] = 0
        self.end_ptr = 0
        self.start_ptr = 0
    
    def store(self, s, a, a_lp, r, s_):

        if self.end_ptr == self.size:
            raise Exception("Buffer is already full")
            
        self.s[self.end_ptr] = s
        self.a[self.end_ptr] = a
        self.a_lp[self.end_ptr] = a_lp
        self.r[self.end_ptr] = r
        self.s_[self.end_ptr] = s_

        self.end_ptr += 1

    def finish_path(self, last_val):
        self.ret[self.end_ptr-1] = self.r[self.end_ptr-1] + self.gamma * last_val
        for idx in reversed(range(self.start_ptr, self.end_ptr-1)):
            self.ret[idx] = self.r[idx] + self.gamma * self.ret[idx+1]
        self.start_ptr = self.end_ptr