from typing import Tuple, List, Callable
import sys
import time
import ctypes
import numpy as np


def find_nearest_idx(arr:np.ndarray, item) -> int:
    '''
    arr: must be sorted
    '''
    return np.abs(arr - item).argmin()


def seq_idx(num_samples:int, num_o:int=1, num_a:int=20, 
            pad_o:bool=True, pad_a:bool=True, pad_aa:bool=True, num_aa:int=10) -> Tuple[List[List[int]], List[List[int]], List[List[bool]], List[List[bool]]]:
    o_idxs, a_idxs = [], []
    pad_o_idts, pad_a_idts = [], []
    for current_idx in range(num_samples - 1):
        selected = True

        o_begin_idx = max(0, current_idx - num_o + 1)
        o_end_idx = min(num_samples, current_idx + 1)
        o_padding = num_o - (o_end_idx - o_begin_idx)
        if o_padding > 0:
            if pad_o:
                o_selected_idxs = [0] * o_padding + list(range(o_begin_idx, o_end_idx))
                o_pad_idts = [True] * o_padding + [False] * (o_end_idx - o_begin_idx)
            else:
                selected = False
        else:
            o_selected_idxs = list(range(o_begin_idx, o_end_idx))
            o_pad_idts = [False] * (o_end_idx - o_begin_idx)

        a_begin_idx = min(num_samples - 1, current_idx + 1)
        a_end_idx = min(num_samples, current_idx + 1 + num_a)
        a_padding = num_a - (a_end_idx - a_begin_idx)
        if a_padding > 0:
            if pad_a and pad_aa:
                if a_padding > num_a - num_aa:
                    selected = False
                else:
                    a_selected_idxs = list(range(a_begin_idx, a_end_idx)) + [num_samples - 1] * a_padding
                    a_pad_idts = [False] * (a_end_idx - a_begin_idx) + [True] * a_padding
            elif pad_a and not pad_aa:
                a_selected_idxs = list(range(a_begin_idx, a_end_idx)) + [num_samples - 1] * a_padding
                a_pad_idts = [False] * (a_end_idx - a_begin_idx) + [True] * a_padding
            else:
                selected = False
        else:
            a_selected_idxs = list(range(a_begin_idx, a_end_idx))
            a_pad_idts = [False] * (a_end_idx - a_begin_idx)
        
        if selected:
            o_idxs.append(o_selected_idxs)
            a_idxs.append(a_selected_idxs)
            pad_o_idts.append(o_pad_idts)
            pad_a_idts.append(a_pad_idts)
    return (o_idxs, a_idxs, pad_o_idts, pad_a_idts)


def loop_timing(interval:float) -> Callable[[], None]:
    """
    Automatically select loop timing strategy based on platform:
    - Windows: absolute timing, timeBeginPeriod optimization, 2ms hybrid wait
    - Linux:  relative timing, 1ms hybrid wait
    interval: update interval second
    """
    
    # Use high-precision clock source uniformly
    time_func = time.perf_counter

    # ============================================
    # Windows platform
    # ============================================
    if sys.platform == 'win32':
        # Windows platform optimization
        try:
            ctypes.windll.winmm.timeBeginPeriod(1)
        except Exception:
            pass

        # Initialize target time
        next_target = time_func() + interval

        def wait_for_next_windows():
            nonlocal next_target
            
            # 1. Calculate remaining time
            remain = next_target - time_func()

            # 2. Hybrid Wait - threshold 0.002s
            if remain > 0.002:
                time.sleep(remain - 0.002)
                remain = next_target - time_func()

            # 3. Busy Wait
            while remain > 0:
                remain = next_target - time_func()

            # 4. Increment absolute time (attempts to catch up even if stalled)
            next_target += interval

        return wait_for_next_windows

    # ============================================
    # Linux/Other platforms
    # ============================================
    else:
        # Default parameter slack_time of the first function
        slack_time = 0.001
        start_time = time_func()

        def wait_for_next_linux():
            nonlocal start_time
            
            # Calculate elapsed time
            elapsed = time_func() - start_time
            sleep_time = max(0, interval - elapsed)
            
            if sleep_time > 0:
                # Here implements the logic of precise_sleep in the original function
                # Sleep for most of the time first
                if sleep_time > slack_time:
                    time.sleep(sleep_time - slack_time)
                
                # Busy wait for remaining time
                while (time_func() - start_time) < interval:
                    pass
            
            # Reset start point to current time (relative timing logic, no catch-up)
            start_time = time_func()

        return wait_for_next_linux


if __name__ == '__main__':
    wait_for_next_iteration = loop_timing(1)

    idx = 0
    while True:
        print(f"Hello, World! {idx}")
        idx += 1

        wait_for_next_iteration()
