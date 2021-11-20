#!/usr/bin/env python

import codelang
import util
import config

import hexdump
import wasmtime

from types import SimpleNamespace as sns

import argparse
import ctypes
import glob
import json
import multiprocessing
import os
import random
import re
import shutil
import signal
import sys
import textwrap
import time


# codeling header, taken from template.wat
HEADER = b'\xfc\x00\x00\x00\x00\x01\x00\x00\x00\x80\x00\x00\x00\xc0\x00\x00'

# signal for end of the 'out' (output) section of Codeling memory
EOF_LEN = 0x10
EOF = b'\x00' * EOF_LEN

# test values for the Codeling.score_LEB128xxx(...) scoring functions
TEST_VALS = (0x0c, 0x1b, 0x3f, 0x40, 0x7f, 0x80, 0x13c, 0x1f4d, 
             0x12345, 0x23456, 0x45678, 0x7f7654,
             0x5c3d123, 0x7ffffff, 0x8000000, 0x7f4bb765)

# set to True by SIGINT_handler() (called when the user presses Ctrl-C)
STOPPING = False


def i32_load(data_ptr, addr: int) -> int:
    return int.from_bytes(data_ptr[addr:addr + 4], 'little')


def i32_store(data_ptr, addr: int, val: int) -> None:
    dst = ctypes.addressof(data_ptr.contents) + addr
    ctypes.memmove(dst, val.to_bytes(4, 'little'), 4)


def bytes_store(data_ptr, addr: int, b: bytes) -> None:
    dst = ctypes.addressof(data_ptr.contents) + addr
    ctypes.memmove(dst, b, len(b))


def all_json_fnames(d: str):
    return sorted(glob.glob(os.path.join(d, '*.json')))


def json2wasm(json_fname: str) -> str:
    return re.sub(r'\.json$', '.wasm', json_fname)


def link2dir(f: str, d: str) -> None:
    try:
        os.link(f, os.path.join(d, os.path.basename(f)))
    except FileExistsError:
        pass


def nice_now():
    return time.strftime('%Y-%m-%d %H:%M:%S %Z', time.localtime())


def nice_now_UTC():
    """
    time.gmtime() sounds like GMT, but it's UTC under the hood:
    https://docs.python.org/3.8/library/time.html#time.gmtime
      " Convert a time expressed in seconds since the epoch to a struct_time
        in UTC in which the dst flag is always zero. "
    """
    return time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())


def comment(s: str) -> str:
    return re.sub(r'^', '# ', s, flags=re.MULTILINE)


def get_release():
    with open('release', 'r') as f:
        return f.readline().strip()


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
        """
        Deferred Initialisation
        - - - - - - - - - - - -
        
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
        """
        XXX TODO UPDATE THIS
        
        See the long comment at the beginning of this file for an overview
        (section "Analysis of template.wasm")
        
        The initial bit is:
        
           0x01 (1 function)
        
        The initial bit in the inner size() means:
        
           0x01 (1 locals entry: ) 0x10 (declare 16 locals of ) 0x7f (type i32)
        
        """
        return self.cfg.template[:0x23] \
            + util.size(b'\x01' + util.size(b'\x01\x10\x7f' + self.b())) \
            + self.cfg.template[0x36:]
    
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
        
        for seg, length in \
                (('rnd', 0x10), ('tmp', 0x40), ('inp', 0x40), ('out', 0x40)):
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
        
        self.json = {
            'ID': ID,
            'code': f.b().hex(),
            'created': nice_now_UTC(),
            'parents': [],
            'created_by': self.cfg.this_script_release + ' Codeling.gen0()'}
    
    def mutate(self, ID: str, L: int, json_source: str) -> bool:
        source_cdl = Codeling(self.cfg, json_fname=json_source)
        targ = 0
        f = codelang.Function(parse=(source_cdl.b(), targ))
        retval = f.mutate(self.cfg.mtfn, L)
        
        desc = f"Codeling.mutate() -length {L} -mtfn {self.cfg.mtfn}"
        self.json = {
            'ID': ID,
            'code': f.b().hex(),
            'created': nice_now_UTC(),
            'parents': [source_cdl.json['ID']],
            'created_by': self.cfg.this_script_release + ' ' + desc}
        
        return retval
    
    def concat(self, cdl: 'Codeling', child_ID: str) -> 'Codeling':
        child = {
            'ID': child_ID,
            'code': re.sub('0b$', '', self.json['code']) + cdl.json['code'],
            'created': nice_now_UTC(),
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
        """Checks that the header is intact and whether there was any output to
        'out'."""
        
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
        return sns(ID=self.json['ID'], score=score, desc=desc + msg, fuel=fuel,
            t_run=t_run)
    
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
                graded_end = 'i' * diff    # out > targ, these are (i)nsertions
                score -= 0x08 * diff
                L = len(targ)
            elif diff < 0:                 # out < targ, these are (d)eletions
                diff = -diff
                graded_end = 'd' * diff
                score -= 0x08 * diff
            
            for o, t in zip(out, targ):
                if o == t:
                    if o == 0x00:
                        graded += 'Z'      # (Z)ero
                        score += 0x20
                    elif o == 0x0b:
                        graded += 'E'      # (E)nd
                        score += 0x40
                    else:
                        graded += 'M'      # (M)atch
                        score += 0x80
                else:
                    graded += 'x'          # substitution
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
        written), -0x40 for overwriting the header or a runtime exception (incl 
        running out of fuel). Rewards: +0x100 when there is any output to 'out' 
        plus +0x80 for every correct non-zero byte written to 'out' (reduced to 
        +0x20 for 0x00 'zero' bytes and +0x40 for 0x0b 'end' bytes). Summed up 
        over 16 test values. A final reward +0x200*(n-1) is awarded for the 
        number n of different output strings across all test values. 
        Description key: i = insertion, d = deletion, x = substition, Z = 
        matching 0x00 zero byte, E = matching 0x0b end byte, M = matching 
        non-zero non-end byte"""
        
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


def score_Codeling(cdl: 'Codeling'):
    return cdl.score()


def score_Codelings(cfg: 'Config', cdl_gtor) -> None:
    IDlen = len(cfg.runid) + 13 if cfg.runid is not None else 20
    times = 't_gen t_valid t_run t_score'.split()
    t_start = time.time()
    
    print('#', ' '.join(sys.argv))
    print('# Release ' + cfg.release,
          'Codeling.score_' + cfg.scfn + '()',
          f"thresh={cfg.thresh:#x}" if cfg.thresh is not None \
              else 'no threshold',
          sep=', ')
    if cfg.thresh is None:
        print("# 'No threshold' means that all codelings will be rejected")
    
    print('# Started:', nice_now())
    print('#')
    print("# All times below are in microseconds, the scores are in hex")
    print(f"{'# ID':{IDlen}}",
          *[f"{s:>7}" for s in \
              times + 'fuel score status'.split() + ['n_acc ']],
          'description',
          sep="\t")
    
    n_scored, n_accepted = 0, 0
    n_scored_prev, t_prev = 0, t_start
    
    with multiprocessing.Pool(cfg.nproc, init_pool_worker) as p:
        # `.imap` because `.map` converts the iterable to a list
        # `_unordered` because don't care about order, really
        # *** when debugging use `map` instead for cleaner error messages ***
        for r in map(score_Codeling, cdl_gtor):
        #for r in p.imap_unordered(score_Codeling, cdl_gtor, chunksize=20):
            n_scored += 1
            if r.status == 'accept':
                n_accepted += 1
            
            micros = [round(1e6 * getattr(r, t)) for t in times]
            print(f"{r.ID:{IDlen}}",
                  *[f"{s:>7}" for s in micros + \
                      [r.fuel, f"{r.score:x}", r.status, f"{n_accepted} "]],
                  r.desc,
                  sep="\t")
            
            if n_scored % 1_000_000 == 0:
                if n_scored == 1_000_000:
                    print(f"{'time':23}",
                          *[f"{s:>15}" for s in \
                              ('n_scored/1e6', 'n_accepted', 'scored/hour')],
                          sep="\t", file=sys.stderr)
                
                t_now = time.time()
                thrpt = (n_scored - n_scored_prev) / (t_now - t_prev) * 3600
                n_scored_prev, t_prev = n_scored, t_now
                print(nice_now(),
                      *[f"{i:>15d}" for i in \
                          (round(n_scored / 1e6), n_accepted)],
                      f"{thrpt:>15.2e}",
                      sep="\t", file=sys.stderr)
    
    print('# Finished:', nice_now())
    print(f"# Throughput: {n_scored/(time.time()-t_start)*3600:.2e} "
          f"scored/hour")


def gtor_score(cfg: 'Config') -> 'Codeling':
    for json_fname in all_json_fnames(cfg.indir):
        yield Codeling(cfg, json_fname=json_fname,
                       wasm_fname=json2wasm(json_fname), deferred=True)


def gtor_gen0(cfg: 'Config', Ls, N: int) -> 'Codeling':
    cdl_i = 0
    for i in range(N):
        for L in Ls:
            ID = f"{cfg.runid}-{cdl_i:012}"
            cdl_i += 1
            yield Codeling(cfg, gen0=(ID, L), deferred=True)


def gtor_mutate(cfg: 'Config', Ls, N: int) -> 'Codeling':
    cdl_i = 0
    for i in range(N):
        for json_fname in all_json_fnames(cfg.indir):
            for L in Ls:
                ID = f"{cfg.runid}-{cdl_i:012}"
                cdl_i += 1
                yield Codeling(cfg, mutate=(ID, L, json_fname), deferred=True)


def stop_check(cdl_gtor) -> 'Codeling':
    for cdl in cdl_gtor:
        if STOPPING:
            print(' Caught SIGINT, stopping. Waiting for jobs to finish.',
                  file=sys.stderr)
            break
        else:
            yield cdl


# TODO XXX uses LastID, needs a rewrite
# also get rid of old 'if STOPPING'
def gtor_concat(cfg: 'Config', gen: int) -> 'Codeling':
    IDs = LastID(write_at_exit=False)
    alive = all_alive_json()
    for fn1 in alive:
        cdl1 = Codeling(fn1)
        gen1 = cdl1.gen()
        print(cdl1.json['ID'], '...')
        
        for fn2 in alive:
            if STOPPING:
                print(' Caught SIGINT, stopping. Waiting for jobs to finish.',
                      file=sys.stderr)
                break
            
            cdl2 = Codeling(fn2)
            gen2 = cdl2.gen()
            
            if gen in (gen1, gen2):
                child_gen = f"{ max(gen1, gen2) + 1 :04x}"
                child_ID = f"{child_gen}-{IDs.next_ID(child_gen):08x}"
                child = cdl1.concat(cdl2, child_ID)
                yield child


def uniq(cfg: 'Config'):
    seen = {}
    
    print(f"# Reading '{cfg.indir}' ...")
    for json_fname in all_json_fnames(cfg.indir):
        cdl = Codeling(cfg, json_fname=json_fname)
        seen[cdl.json['code']] = json_fname
    
    print(f"# Making '{cfg.outdir}' unique ...")
    n_files = 0
    for json_fname in all_json_fnames(cfg.outdir):
        cdl = Codeling(cfg, json_fname=json_fname)
        code = cdl.json['code']
        if code in seen:
            print(f"'{json_fname}' has same code as '{seen[code]}', deleting")
            os.remove(json_fname)
            os.remove(json2wasm(json_fname))
        else:
            seen[code] = json_fname
            n_files += 1
    
    print(f"# {n_files} files left in '{cfg.outdir}'")


def LEB128_test():
    for i32 in (0x00, 0x3f, 0x40, 0x7f, 0x80, 0xbf, 0xc0, 0x1fff, 0x2000,
                0x7ffffff, 0x8000000,
                0x7fffffff, 0x80000000, 0xf7ffffff, 0xf8000000,
                0xffffffbf, 0xffffffc0, 0xffffffff,
                *TEST_VALS):
        
        b = util.LEB128(util.unsigned2signed(i32, 32))
        bs = util.ByteStream(b)
        parsed = util.signed2unsigned(bs.next_LEB128(), 32)
        assert parsed == i32
        print(f"i32={hex(i32)}  b=0x{b.hex()}  parsed={hex(parsed)}")


def hack():
    LEB128_test()


def print_item_desc(content):
    item_w = max([len(item) for item, _ in content])
    desc_w = shutil.get_terminal_size().columns - item_w - 6
    
    for item, desc in content:
        desc = ' '.join(desc.split())  # get rid of multiple spaces and \n
        wrapped = textwrap.wrap(desc, width=desc_w)
        for i, d in zip([item] + [''] * len(wrapped), wrapped):
            print(f"  {i:{item_w}}  {d}")


def list_available_scfns():
    print("Available scoring functions:")
    
    fns = []
    for fn in dir(Codeling):
        m = re.match("score_(.*)", fn)
        if m:
            fn_code = m.group(1)
            doc = getattr(getattr(Codeling, fn), '__doc__')
            fns.append((fn_code, doc))
    
    print_item_desc(fns)


def list_available_mtfns():
    print("Available mutator functions:")
    
    fns = []
    for fn in dir(codelang.Function):
        m = re.match("mutator_(.*)", fn)
        if m:
            fn_code = m.group(1)
            doc = getattr(getattr(codelang.Function, fn), '__doc__')
            fns.append((fn_code, doc))
    
    print_item_desc(fns)


def main():
    """Generate, mutate and score codelings"""
    
    cfg = config.Config()
    cmd = sys.argv[0]
    epilogue = f"""\
        Most of the options can also be set in `config.py`.
        
        Examples:
          # print out a list of scoring functions and their descriptions
          {cmd} -scfn list
          
          # generate 10 new generation 0 codelings of default length and print
          # out their scores
          {cmd} -gen0 10 run01
          
          # generate 10 strings of random bytes of length 5, score them with
          # scoring function 'Codelings.score_v02()', print out their scores
          # and save those with score >= 0x00 to '{cfg.outdir}'
          {cmd} -rnd0 10 -length 5 -scfn v02 -thresh 0x00 run02
          
        Feel free to use Ctrl-C to gracefully end the script at any point.
        """
    
    defaults = (
        ('length', 5),
        ('fuel', 50),
        ('scfn', 'LEB128_nolen'),
        ('mtfn', 'ins'),
        ('thresh', None),
        ('nproc', multiprocessing.cpu_count()),
        ('template_file', 'template.wasm'))
    
    # if the user has already set the option in Config.py, keep their value
    for attr, val in defaults:
        if not hasattr(cfg, attr):
            setattr(cfg, attr, val)
    
    default_T = 'no codelings are saved' if cfg.thresh is None else cfg.thresh
    
    cfg.release = get_release()
    cfg.this_script = os.path.basename(__file__)
    cfg.this_script_release = f"{cfg.this_script} v{cfg.release}"
    
    # used internally for hacking on new features
    if len(sys.argv) > 1 and sys.argv[1] == '-hack':
        hack()
        return
    
    # hard-coded to pre-empt checking for required options
    for i in range(2, len(sys.argv)):
        if sys.argv[i] == 'list':
            if sys.argv[i-1] == '-scfn':
                list_available_scfns()
                return
            elif sys.argv[i-1] == '-mtfn':
                list_available_mtfns()
                return
    
    def type_int_ish(s: str):
        try:
            return int(s, 0)
        except ValueError:
            try:
                return round(float(s))
            except ValueError:
                raise argparse.ArgumentTypeError(
                    f"failed to convert '{s}' to an int, try something like "
                    "'123', '12e3' or '0xf0'.")
    
    def type_scfn(s: str):
        fn = 'score_' + s
        if hasattr(Codeling, fn):
            return s
        else:
            raise argparse.ArgumentTypeError(f"there is no '{fn}' in Codeling")
    
    def type_mtfn(s: str):
        fn = 'mutator_' + s
        if hasattr(codelang.Function, fn):
            return s
        else:
            raise argparse.ArgumentTypeError\
                (f"there is no '{fn}' in codelang.Function")
    
    def type_dir_str(s: str):
        if os.path.isdir(s):
            return s
        else:
            raise argparse.ArgumentTypeError(f"'{s}' is not a directory")
    
    parser = argparse.ArgumentParser(
        description=main.__doc__, epilog=textwrap.dedent(epilogue),
        formatter_class=argparse.RawDescriptionHelpFormatter)
    
    cmds = parser.add_mutually_exclusive_group(required=True)
    cmds.add_argument('-score', action='store_true',
        help=f"""Score all codelings in '{cfg.indir}'.""")
    cmds.add_argument('-rnd0', type=type_int_ish, metavar='N',
        help="""Generate N new random generation 0 codelings that are 
        completely random strings of bytes. Most codelings generated in this 
        way will not pass the WebAssembly validation step (syntax check).""")
    cmds.add_argument('-gen0', type=type_int_ish, metavar='N',
        help="""Generate N new random generation 0 codelings using the Reduced 
        Instruction Set (see 'progress_so_far.md' for details XXX TODO). All 
        codelings generated in this way will pass the WebAssembly validation 
        step (syntax check).""")
    cmds.add_argument('-mutate', type=type_int_ish, metavar='N',
        help=f"""Generate new codelings by mutating each codeling in
        '{cfg.indir}' N times.""")
    cmds.add_argument('-concat', type=type_int_ish, metavar='gen',
        help=f"""For all codelings X and Y in '{cfg.indir}' such that at least 
        one is of generation 'gen' (eg '0' or '0x00'), create a new codeling 
        X+Y by concatenating their codes. XXX TODO BROKEN""")
    cmds.add_argument('-uniq', action='store_true',
        help=f"""Look at all codelings in '{cfg.outdir}' and remove those that 
        are duplicates of (i.e. have the same code as) codelings in 
        '{cfg.indir}' or earlier codelings in '{cfg.outdir}'.""")
    
    parser.add_argument('-upto', action='store_true',
        help=f"""For options that use the length parameter L (see '-length' 
        below for details), equivalent to repeatedly running this script with 
        '-length 1', '-length 2', ..., '-length L' (e.g. '-upto -length 5').""")
    parser.add_argument('-length', type=type_int_ish, metavar='L',
        help=f"""For options that produce new codelings, set the 'length' 
        parameter to L. This parameter has slightly different meanings 
        depending on which method for generating or mutating codelings is used: 
        for -rnd0 it is the length of the new random strings in bytes, for 
        -gen0 it is the minimum number of instructions in the new codelings, 
        and for -mutate is the minimum number of instructions changed in the 
        parent to produce the new codelings. (Default: {cfg.length})""")
    parser.add_argument('-fuel', type=type_int_ish, metavar='F', 
        help=f"""When running a WebAssembly function, provide it with F units 
        of fuel. This limits the number of instructions that will be executed 
        before the function runs out of fuel, thereby preventing infinite 
        loops. [From `store.rs` in the Wasmtime sources: "Most WebAssembly 
        instructions consume 1 unit of fuel. Some instructions, such as `nop`, 
        `drop`, `block`, and `loop`, consume 0 units, as any execution cost 
        associated with them involves other instructions which do consume 
        fuel."] (Default: {cfg.fuel})""")
    parser.add_argument('-scfn', type=type_scfn, metavar='name',
        help=f"""Scoring function to use, e.g. 'v02' for 
        'Codeling.score_v02()'. '-scfn list' lists all available scoring
        functions along with their descriptions. (Default: '{cfg.scfn}')""")
    parser.add_argument('-mtfn', type=type_mtfn, metavar='name',
        help=f"""Mutator function to use, e.g. 'ins' for 
        'codelang.Function.mutator_ins()'. '-mtfn list' lists all available
        mutator functions along with their descriptions. (Default:
        '{cfg.mtfn}')""")
    parser.add_argument('-thresh', type=type_int_ish, metavar='T',
        help=f"""Codelings with score >= T (e.g. '0x5f') are saved to
        '{cfg.outdir}'. For existing codelings (e.g. those taken from
        '{cfg.indir}') this creates hard links to the originals. If you ever
        want to use a negative threshold, try '" -0x40"' - note the quotation
        marks and the initial space. (Default: {default_T})""")
    parser.add_argument('-indir', type=type_dir_str, metavar='path',
        help=f"""Change the input directory to 'path'. (Default: 
        '{cfg.indir}')""")
    parser.add_argument('-outdir', type=type_dir_str, metavar='path',
        help=f"""Change the output directory to 'path'. (Default:
        '{cfg.outdir}')""")
    parser.add_argument('-nproc', type=type_int_ish, metavar='N', 
        help=f"""Set the number of worker processes to use in the pool to N. 
        (Default for this machine: {cfg.nproc})""")
    
    parser.add_argument('runid', type=str, nargs='?',
        help=f"""Identifier for this run (e.g. 'run01'). All codelings produced 
        during the run will have identifiers of the form 'runid-012345678901', 
        i.e. the run identifier followed by a dash and twelve digits (with most 
        of the leading ones being zero). The run identifier is a required 
        argument for all types of runs except '-score' and '-uniq' where no new 
        codelings are produced.""")
    
    args = parser.parse_args()
    
    new_cdls = (args.rnd0, args.gen0, args.mutate, args.concat)
    if any([a is not None for a in new_cdls]) and args.runid is None:
        parser.error("'runid' is required for all runs except " \
                     "'-score' and '-uniq'")
    
    if args.runid is not None and re.match(r'^#', args.runid):
        parser.error("'runid' cannot start with '#'")
    
    l = 'length fuel scfn mtfn thresh indir outdir nproc runid'
    for param in l.split():
        a = getattr(args, param)
        if a is not None or param == 'runid':
            setattr(cfg, param, a)
    
    if args.upto:
        Ls = range(1, cfg.length + 1)
    else:
        Ls = (cfg.length,)
    
    gtor = None
    
    if args.score:
        gtor = gtor_score(cfg)
    elif args.rnd0 is not None:
        sys.exit("SORRY, -rnd0 not implemented yet (well, re-implemented) :-(")
    elif args.gen0 is not None:
        gtor = gtor_gen0(cfg, Ls, args.gen0)
    elif args.mutate is not None:
        gtor = gtor_mutate(cfg, Ls, args.mutate)
    elif args.concat is not None:
        gtor = gtor_concat(cfg, args.concat)
    elif args.uniq:
        uniq(cfg)
        return
    
    with open(cfg.template_file, "rb") as f:
        cfg.template = f.read()
    
    signal.signal(signal.SIGINT, SIGINT_handler)
    score_Codelings(cfg, stop_check(gtor))


def SIGINT_handler(sig, frame):
    global STOPPING  # signal handler, so needs to be explicitly declared
    STOPPING = True


def init_pool_worker():
    # ignore SIGINT in pool workers
    signal.signal(signal.SIGINT, signal.SIG_IGN)


if __name__ == "__main__":
    main()
