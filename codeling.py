import codelang
import util

import hexdump
import wasmtime

from types import SimpleNamespace as sns

import ctypes
import json
import re
import os
import time


# codeling header, taken from template.wat
HEADER = b'\xfc\x00\x00\x00\x00\x01\x00\x00\x00\x80\x00\x00\x00\xc0\x00\x00'

# signal for end of the 'out' (output) section of Codeling memory
EOF_LEN = 0x10
EOF = b'\x00' * EOF_LEN


def i32_load(data_ptr, addr: int) -> int:
    return int.from_bytes(data_ptr[addr:addr + 4], 'little')


def i32_store(data_ptr, addr: int, val: int) -> None:
    dst = ctypes.addressof(data_ptr.contents) + addr
    ctypes.memmove(dst, val.to_bytes(4, 'little'), 4)


def bytes_store(data_ptr, addr: int, b: bytes) -> None:
    dst = ctypes.addressof(data_ptr.contents) + addr
    ctypes.memmove(dst, b, len(b))


def comment(s: str) -> str:
    return re.sub(r'^', '# ', s, flags=re.MULTILINE)


class Codeling:
    def __init__(
            self,
            cfg: 'Config',
            json_fname: str = None,
            wasm_fname: str = None,
            json: dict = None,
            gen0=None,
            mutate=None,
            deferred=False):
        """Deferred Initialisation
        
        deferred=True is used by the parent process to shift most of the load
        onto child processes in the pool. When Codeling.score() is called from
        the child, it runs self._deferred_init() as its first step and
        initialisation takes place at that point.
        """
        if deferred:
            self._deferred = locals()
            self._deferred['deferred'] = False
            del(self._deferred['self'])
            return
        else:
            self._deferred = None
        
        self.cfg = cfg
        
        config = wasmtime.Config()
        config.consume_fuel = True
        self.store = wasmtime.Store(wasmtime.Engine(config))
        
        # file names we have actually read from or written to (none so far)
        self._json_fname = None
        self._wasm_fname = None
        
        # ... but save the param we've been passed
        # NB read_wasm() is an important part of scoring (validation step)
        # and takes place at that point
        self._init_wasm_fname = wasm_fname
        
        if sum(x is not None for x in (json_fname, json, gen0, mutate)) != 1:
            raise RuntimeError("need exactly one of these params: "
                "(json_fname, json, gen0, mutate)")
        
        if json_fname is not None:
            self.read_json(json_fname)
        elif json is not None:
            self.json = json
        elif gen0 is not None:
            self.gen0(*gen0)
        elif mutate is not None:
            if not self.mutate(*mutate):
                raise RuntimeError('Failed to generate codeling')
    
    def _deferred_init(self) -> None:
        if self._deferred:
            self.__init__(**self._deferred)
    
    def b(self) -> bytes:
        return bytes.fromhex(self.json['code'])
    
    def write_json(self, outdir: str) -> str:
        if 'json_version' not in self.json:
            self.json['json_version'] = 1
        
        fname = os.path.join(outdir, self.json['ID'] + '.json')
        with open(fname, 'w') as outf:
            outf.write(json.dumps(self.json, indent=4))
        
        self._json_fname = fname
        return fname

    def read_json(self, fname: str):
        if not fname.endswith('.json'):
            raise RuntimeError("'fname' needs to end in '.json'")
        
        with open(fname, 'r') as f:
            self.json = json.loads(f.read())
        
        if 'json_version' not in self.json:
            self.json['json_version'] = 1
        
        self._json_fname = fname
    
    def wasm_bytes(self) -> bytes:
        """XXX TODO UPDATE THIS
        
        See the long comment at the beginning of this file for an overview
        (section "Analysis of template.wasm")
        
        The initial bit is:
        
           0x01 (1 function)
        
        The initial bit in the inner size() means:
        
           0x01 (1 locals entry: ) 0x10 (declare 16 locals of) 0x7f (type i32)
        
        """
        return (self.cfg.template[:0x23] +
                util.size(b'\x01' + util.size(b'\x01\x10\x7f' + self.b())) +
                self.cfg.template[0x36:])
    
    def write_wasm(self, wasm_bytes: bytes, outdir: str) -> str:
        fname = os.path.join(outdir, self.json['ID'] + '.wasm')
        if not os.path.exists(fname):
            with open(fname, "wb") as f:
                f.write(wasm_bytes)
        
        self._wasm_fname = fname
        return fname
    
    def update_mem(self):
        self.mem_ptr = self.mem.data_ptr(self.store)
        self.mem_len = self.mem.data_len(self.store)
    
    def create_instance(self, module: 'wasmtime.Module'):
        instance = wasmtime.Instance(self.store, module, [])
        exports = instance.exports(self.store)
        self.mem = exports['m']
        assert type(self.mem) is wasmtime.Memory
        self.update_mem()
        
        self._f = exports['f']
        assert type(self._f) is wasmtime.Func
        
        self.rnd_addr, self.tmp_addr, self.inp_addr, self.out_addr = \
            [i32_load(self.mem_ptr, addr) for addr in range(0x00, 0x10, 0x04)]
        
        self._instance = instance
    
    def read_wasm(self, fname: str):
        if not fname.endswith('.wasm'):
            raise RuntimeError("'fname' needs to end in '.wasm'")
        
        module = wasmtime.Module.from_file(self.store.engine, fname)
        self.create_instance(module)
        self._wasm_fname = fname
    
    def from_bytes(self, b: bytes):
        self.create_instance(wasmtime.Module(self.store.engine, b))
    
    def link_json_wasm(self, outdir) -> None:
        for f in (self._json_fname, self._wasm_fname):
            os.link(f, os.path.join(outdir, os.path.basename(f)))
    
    def instance_info(self) -> None:
        d = self._instance.exports(self.store)._extern_map
        mems = [key for key, val in d.items() if type(val) is wasmtime.Memory]
        print(f'memories exported: {mems}')
        
        funs = [key for key, val in d.items() if type(val) is wasmtime.Func]
        print(f'functions exported: {funs}')
        
        size = self.mem.size(self.store)
        print(f'mem: pages=0x{size:x} bytes=0x{self.mem_len:x}')
        print(f'rnd addr=0x{self.rnd_addr:06x}')
        print()
    
    def memdump(self) -> None:
        print('dumping memory')
        print('- hdr @ 0x0000:')
        hexdump.hexdump(bytearray(self.mem_ptr[:0x10]))
        
        for seg, length in (('rnd', 0x10), ('tmp', 0x40),
                            ('inp', 0x40), ('out', 0x40)):
            addr = getattr(self, seg + '_addr')
            start = addr & 0xfff0
            print(f'- {seg} @ 0x{addr:04x}:')
            hexdump.hexdump(bytearray(self.mem_ptr[start: start + length]))
            print('   :')
        
        print()
    
    def reset_hdr(self) -> None:
        bytes_store(self.mem_ptr, 0x00, HEADER)
    
    def set_rnd(self) -> None:
        i32_store(self.mem_ptr, self.rnd_addr, random.getrandbits(32))
    
    def set_inp(self, b: bytes) -> None:
        bytes_store(self.mem_ptr, self.inp_addr, b)
    
    def clear_out(self) -> None:
        bytes_store(self.mem_ptr, self.out_addr, b'\x00'*0x100)
    
    def get_out(self) -> bytes:
        zero_len = 0
        for i in range(self.out_addr, self.mem_len):
            if self.mem_ptr[i] == 0x00:
                zero_len += 1
                if zero_len == EOF_LEN:
                    return bytes(self.mem_ptr[self.out_addr: i-EOF_LEN+1])
            else:
                zero_len = 0
        
        return bytes(self.mem_ptr[self.out_addr: self.mem_len])
    
    def gen0(self, ID: str, L: int) -> None:
        targ = 0
        f = codelang.Function(gen0=(targ, L))
        
        desc = f"Codeling.gen0() -length {L}" 
        self.json = {
            'ID': ID,
            'code': f.b().hex(),
            'created': util.nice_now_UTC(),
            'parents': [],
            'created_by': self.cfg.this_script_release + ' ' + desc}
    
    def mutate(self, ID: str, L: int, json_source: str) -> bool:
        source_cdl = Codeling(self.cfg, json_fname=json_source)
        targ = 0
        f = codelang.Function(parse=(source_cdl.b(), targ))
        retval = f.mutate(self.cfg.mtfn, L)
        
        desc = f"Codeling.mutate() -length {L} -mtfn {self.cfg.mtfn}"
        self.json = {
            'ID': ID,
            'code': f.b().hex(),
            'created': util.nice_now_UTC(),
            'parents': [source_cdl.json['ID']],
            'created_by': self.cfg.this_script_release + ' ' + desc}
        
        return retval
    
    def concat(self, cdl: 'Codeling', child_ID: str) -> 'Codeling':
        child = {
            'ID': child_ID,
            'code': re.sub('0b$', '', self.json['code']) + cdl.json['code'],
            'created': util.nice_now_UTC(),
            'parents': [self.json['ID'], cdl.json['ID']],
            'created_by': self.cfg.this_script_release + ' Codeling.concat()'}
        
        return Codeling(json=child)
    
    def run_wasm(self) -> (str, int, float):
        t_run = time.time()
        
        # JFC, why don't they just let us set the fuel level?
        # Or not die when we ask it how much fuel is left?
        # This weird stuff is needed because it looks as though the fuel level 
        # can go negative (!), in which case consume_fuel(0) throws an error
        # *even after* the add_fuel(...) call.
        fuel_left = 0
        while fuel_left != self.cfg.fuel:
            if fuel_left < self.cfg.fuel:
                self.store.add_fuel(self.cfg.fuel - fuel_left)
                try:
                    fuel_left = self.store.consume_fuel(0)
                except wasmtime.WasmtimeError:
                    fuel_left = 0
            else:
                fuel_left = self.store.consume_fuel(fuel_left - self.cfg.fuel)
        
        e = None
        try:
            # function 'f' exported by the WebAssembly module
            self._f(self.store)
        except Exception as err:
            e = comment(str(err))
        
        self.update_mem()
        return (e, self.store.fuel_consumed(), time.time() - t_run)
    
    def score_v00(self) -> 'SimpleNamespace':
        """Checks whether there was any output to memory at all. WARNING: This
        function is SLOW."""
        
        self.set_rnd()
        self.set_inp(self.b())
        mem = self.mem_ptr[:self.mem_len]
        e, fuel, t_run = self.run_wasm()
        
        score, desc = 0x00, 'OK'
        if e:
            score, desc = -0x40, 'runtime exception\n' + e
        elif mem == self.mem_ptr[:self.mem_len]:
            score, desc = -0x20, 'no output at all'
        
        return sns(ID=self.json['ID'], score=score, desc=desc, fuel=fuel,
                   t_run=t_run)
    
    def score_v01(self) -> 'SimpleNamespace':
        """Checks that the header is intact and whether there was any output 
        to 'out'."""
        
        self.set_rnd()
        self.set_inp(self.b())
        mem = self.mem_ptr[:self.rnd_addr]
        e, fuel, t_run = self.run_wasm()
        
        if e:
            score, desc = -0x40, 'runtime exception\n' + e
        elif mem[:self.rnd_addr] != self.mem_ptr[:self.rnd_addr]:
            score, desc = -0x30, 'overwrote header'
        else:
            out = self.get_out()
            
            if len(out) == 0:
                score, desc = -0x20, 'no output to out'
            else:
                writes = [i + self.out_addr for i, b in enumerate(out) if b]
                score = 0x30 * len(writes) - int(len(self.json['code']) / 2)
                desc = 'OK - ' + ' '.join([f'{w:4x}' for w in writes])
        
        return sns(ID=self.json['ID'], score=score, desc=desc, fuel=fuel,
                   t_run=t_run)
    
    def score_v02(self) -> 'SimpleNamespace':
        """Penalties: -0x01 for every byte of code length, -0x10 for
        overwriting the header or a runtime exception (incl running out of
        fuel), -0x40 when no output to 'out'. Reward: +0x40 for every byte
        written to 'out'."""
        
        self.set_rnd()
        self.set_inp(self.b())
        mem = self.mem_ptr[:self.rnd_addr]
        e, fuel, t_run = self.run_wasm()
        
        score, msg = 0x00, ''
        if e:
            score -= 0x10
            
            if e == '# all fuel consumed by WebAssembly':
                desc = 'out of fuel'
            else:
                desc, msg = 'runtime exception', '\n' + e
        else:
            desc = 'OK'
        
        if mem[:self.rnd_addr] != self.mem_ptr[:self.rnd_addr]:
            score -= 0x10
            desc += ', overwrote header'
        
        out = self.get_out()
        if len(out) == 0:
            score -= 0x40
            desc += ', no output to out'
        else:
            writes = [i + self.out_addr for i, b in enumerate(out) if b]
            score += 0x40 * len(writes)
            desc += ' - ' + ' '.join([f'{w:4x}' for w in writes])
        
        score -= round(len(self.json['code']) / 2)
        return sns(ID=self.json['ID'], score=score, desc=desc + msg,
                   fuel=fuel, t_run=t_run)
    
    def _score_LEB128_val(self, val: int, lenpen: int) -> 'SimpleNamespace':
        self.reset_hdr()
        self.set_inp(val.to_bytes(4, 'little'))
        self.clear_out()
        
        targ = util.LEB128(util.unsigned2signed(val, 32)) + b'\x0b'
        mem = self.mem_ptr[:self.rnd_addr]
        e, fuel, t_run = self.run_wasm()
        
        score, msg = 0x00, ''
        if e:
            score -= 0x40
            
            if e == '# all fuel consumed by WebAssembly':
                desc = 'fuel'
            else:
                desc = 'exc'
        else:
            desc = 'OK'
        
        if mem[:self.rnd_addr] != self.mem_ptr[:self.rnd_addr]:
            score -= 0x40
            desc += ',hdr'
        
        out = self.get_out()
        if len(out) == 0:
            desc += ':-'
        else:
            score += 0x200
            graded, graded_end = '', ''
            L = len(out)
            diff = L - len(targ)
            if 0 < diff:
                graded_end = 'i' * diff # out > targ, these are (i)nsertions
                score -= 0x08 * diff
                L = len(targ)
            elif diff < 0:              # out < targ, these are (d)eletions
                diff = -diff
                graded_end = 'd' * diff
                score -= 0x08 * diff
            
            for o, t in zip(out, targ):
                if o == t:
                    if o == 0x00:
                        graded += 'Z'   # (Z)ero
                        score += 0x20
                    elif o == 0x0b:
                        graded += 'E'   # (E)nd
                        score += 0x40
                    else:
                        graded += 'M'   # (M)atch
                        score += 0x80
                else:
                    graded += 'x'       # substitution
                    score -= 0x08
                    
            desc += ':' + graded + graded_end
        
        score -= lenpen * round(len(self.json['code']) / 2)
        return sns(ID=self.json['ID'], score=score, desc=desc, fuel=fuel,
            t_run=t_run, out=out)
    
    def _score_LEB128(self, lenpen: int) -> 'SimpleNamespace':
        total = sns(ID=self.json['ID'], score=0, desc='', fuel=0, t_run=0)
        desc = []
        outs = set()
        vals = (0x0c, 0x1b, 0x3f, 0x40, 0x5b, 0x7f, 0x80, 0x9f, 
                0xac, 0x12345, 0x23456, 0x45678,
                0x5c3d123, 0x7ffffff, 0x8000000, 0x94bb765)
        
        for val in vals:
            retval = self._score_LEB128_val(val, lenpen)
            total.score += retval.score
            total.fuel = retval.fuel
            total.t_run += retval.t_run
            desc.append(retval.desc)
            outs.add(retval.out.hex())
        
        n = len(outs)
        total.score += 0x100 * (n-1)
        total.desc = f"n={n} " + ' '.join(desc)
        return total
    
    def score_LEB128(self) -> 'SimpleNamespace':
        """Penalties: -0x01 for every byte of code length, -0x08 for every 
        wrong byte written to 'out' (or missing when it should have been 
        written), -0x40 for overwriting the header or a runtime exception 
        (incl running out of fuel). Rewards: +0x100 when there is any output 
        to 'out' plus +0x80 for every correct non-zero byte written to 'out' 
        (reduced to +0x20 for 0x00 'zero' bytes and +0x40 for 0x0b 'end' 
        bytes). Summed up over 16 test values. A final reward +0x200*(n-1) is 
        awarded for the number n of different output strings across all test 
        values. Description key: i = insertion, d = deletion, x = substition, 
        Z = matching 0x00 zero byte, E = matching 0x0b end byte, M = matching 
        non-zero non-end byte
        """
        return self._score_LEB128(lenpen=1)
    
    def score_LEB128_nolen(self) -> 'SimpleNamespace':
        """Same as LEB128 except no penalty for code length"""
        
        return self._score_LEB128(lenpen=0)
    
    def score(self) -> 'SimpleNamespace':
        # generation
        t_gen = time.time()
        
        try:
            self._deferred_init()
        except RuntimeError:
            return sns(ID=self.json['ID'],
                t_gen=(time.time() - t_gen), t_valid=0, t_run=0, t_score=0,
                fuel=0, score=-0xff, status='reject', desc='GENERATION ERROR')
        
        wasm_fname = self._init_wasm_fname
        if wasm_fname is None:
            in_memory = True
            wasm_bytes = self.wasm_bytes()
        else:
            in_memory = False
        
        # validation
        t_valid = time.time()
        try:
            if in_memory:
                self.from_bytes(wasm_bytes)
            else:
                self.read_wasm(wasm_fname)
        except wasmtime.WasmtimeError as e:
            t_score = time.time()
            res = sns(ID=self.json['ID'], score=-0x80,
                      desc='VALIDATION ERROR\n' + comment(str(e)),
                      fuel=0, t_run=0.0)
        else:
            # scoring (includes running)
            t_score = time.time()
            score_fn = getattr(self, 'score_' + self.cfg.scfn)
            res = score_fn()
        
        if self.cfg.thresh is not None and res.score >= self.cfg.thresh:
            res.status = 'accept'
            
            if in_memory:
                self.write_wasm(wasm_bytes, self.cfg.outdir)
            else:
                link2dir(self._wasm_fname, self.cfg.outdir)
            
            if self._json_fname is None:
                self.write_json(self.cfg.outdir)
            else:
                link2dir(self._json_fname, self.cfg.outdir)
        else:
            res.status = 'reject'
        
        res.t_gen = t_valid - t_gen
        res.t_valid = t_score - t_valid
        res.t_score = (time.time() - t_score) - res.t_run
        return res
