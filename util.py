# utility functions shared by `evolver.py` and `code.py`

import random

class ByteStream:
    def __init__(self, b: bytes):
        self.ba = bytearray(b)
        self.ba_ptr = 0
    
    def done(self) -> bool:
        return len(self.ba) <= self.ba_ptr
    
    def next_i8(self) -> int:
        i = self.ba[self.ba_ptr]
        self.ba_ptr += 1
        return i
    
    def next_b(self) -> bytes:
        return self.next_i8().to_bytes(1, 'little')
    
    def next_uLEB128(self) -> int:
        """unsigned Little Endian Base 128 compression for integers, decoder"""
        
        val = 0
        n = 0
        while True:
            i = self.next_i8()
            val += (i & 0x7f) << (7*n)
            n += 1
            
            if i < 0x80:
                return val


def uLEB128(val: int) -> bytes:
    """unsigned Little Endian Base 128 compression for integers, encoder"""
    
    if val > 0:
        out = b''
        while val > 0:
            cur = val & 0x7f
            val >>= 7
            
            if val > 0:
                cur |= 0x80
            
            out += cur.to_bytes(1, 'little')
        
        return out
    elif val == 0:
        return b'\x00'
    else:
        raise ValueError('ERROR: val < 0')


def unsigned2signed(val: int, n_bits: int) -> int:
    thresh = 1 << (n_bits-1)
    c = thresh << 1
    return val if val < thresh else val - c


def LEB128(val: int) -> bytes:
    """(signed) Little Endian Base 128 compression for integers, encoder"""
    
    if val > 0:
        out = b''
        cur = 0x80
        while cur & 0x80:
            cur = val & 0x7f
            val >>= 7
            
            # the `cur & 0x40` is so the highest encoded bit isn't 1
            if val > 0 or cur & 0x40:
                cur |= 0x80
            
            out += cur.to_bytes(1, 'little')
        
        return out
    elif val < 0:
        n_bytes = (val+1).bit_length()//7 + 1
        mask = (1 << 7*n_bytes) - 1
        return uLEB128(val & mask)
    else:
        return b'\x00'


def size(b: bytes) -> bytes:
    return uLEB128(len(b)) + b


def rnd_int(n_bytes: int) -> int:
    return random.getrandbits(8 * n_bytes)


def rnd_bytes(n_bytes: int) -> bytes:
    return rnd_int(n_bytes).to_bytes(n_bytes, 'big')


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
