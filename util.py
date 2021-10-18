# utility functions shared by `evolver.py` and `code.py`

import random

def uLEB128(val: int) -> bytes:
    """unsigned Little Endian Base 128 compression for integers"""
    
    if val == 0:
        return b'\x00'
    elif val < 0:
        raise ValueError('ERROR: val < 0')
    else:
        out = b''
        
        while val > 0:
            cur = val & 0x7f
            val = val >> 7
            
            if val > 0:
                cur |= 0x80
            
            out += cur.to_bytes(1, 'little')
        
        return out


def size(b: bytes) -> bytes:
    return uLEB128(len(b)) + b


def rnd_int(n_bytes: int) -> int:
    return random.getrandbits(8 * n_bytes)


def rnd_bytes(n_bytes: int) -> bytes:
    return random.getrandbits(8 * n_bytes).to_bytes(n_bytes, 'big')


_a = [x for x in range(1, 33)]          # 1 ... 32
_w = [2**x for x in range(31, -1, -1)]  # 2^31 ... 2^0

def rnd_i32_0s1s() -> int:
    """
    Returns mostly 0s and 1s, some 2s and 3s, etc
    
    Value  Probability      How P was calculated
    =====  ===============  ==================================================
    0-1    1/3 each         1/2**2 + 1/4**2 + 1/8**2 + 1/16**2 + 1/32**2 + ...
    2-3    1/(3*4) each     1/4**2 + 1/8**2 + 1/16**2 + 1/32**2 + ...
    4-7    1/(3*4**2) each  1/8**2 + 1/16**2 + 1/32**2 + ...
    8-15   1/(3*4**3) each   :
     :      :
    
    The probabilities sum up to 1 (well almost, NB not an infinite sequence!).
    
    NOTE: Inverse transform sampling might be quicker for the 1st step (inner 
    call to random.choices()), ought to implement both and compare speeds, see 
    the 2nd example in:
    
    https://en.wikipedia.org/wiki/Inverse_transform_sampling#Examples
    """
    return random.getrandbits(random.choices(_a, weights=_w, k=1)[0])