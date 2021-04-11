#!/usr/bin/env python

from Config import Config as cfg

from hexdump import hexdump
from wasmtime import Config, Engine, Store, Module, Instance, Memory, Func, \
	WasmtimeError

from types import SimpleNamespace as sns
import argparse, ctypes, glob, json, multiprocessing, os, random, re, shutil, \
	signal, sys, textwrap, time


# signal for end of the 'out' (output) section of Codeling memory
EOF_LEN = 0x10
EOF = b'\x00' * EOF_LEN

# set to True by SIGINT_handler() (called when the user presses Ctrl-C)
STOPPING = False

def uLEB128(val:int) -> bytes:
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

def size(b:bytes) -> bytes:
	return uLEB128(len(b)) + b

def rnd_int(n_bytes:int) -> int:
	return random.getrandbits(8*n_bytes)

def rnd_bytes(n_bytes:int) -> bytes:
	return random.getrandbits(8*n_bytes).to_bytes(n_bytes,'big')

aux_a = [x for x in range(1,33)]         # 1 ... 32
aux_w = [2**x for x in range(31,-1,-1)]  # 2^31 ... 2^0

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
	
	Inverse transform sampling might be quicker for the 1st step, ought to 
	implement both and compare speeds - see the 2nd example in:
	
	https://en.wikipedia.org/wiki/Inverse_transform_sampling#Examples
	"""
	return random.getrandbits( random.choices(aux_a, weights=aux_w, k=1)[0] )

def i32_load(mem:Memory, addr:int) -> int:
	return int.from_bytes(mem.data_ptr[addr:addr+4], 'little')

def i32_store(mem:Memory, addr:int, val:int) -> None:
	dst = ctypes.addressof(mem.data_ptr.contents) + addr
	ctypes.memmove(dst, val.to_bytes(4, 'little'), 4)

def bytes_store(mem:Memory, addr:int, b:bytes) -> None:
	dst = ctypes.addressof(mem.data_ptr.contents) + addr
	ctypes.memmove(dst, b, len(b))

def all_json_fnames(d: str):
	return sorted(glob.glob(os.path.join(d, '*.json')))

def json2wasm(json_fname: str) -> str:
	return re.sub(r'\.json$', '.wasm', json_fname)

def link2dir(f:str, d:str) -> None:
	os.link(f, os.path.join(d, os.path.basename(f)))

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

def comment(s:str) -> str:
	return re.sub(r'^', '# ', s, flags=re.MULTILINE)

def get_release():
	with open('release', 'r') as f:
		return f.readline().strip()


class Code:
	"""
	Generate short random codes that pass WebAssembly validation
	
	Used by the `-gen0` option in main()
	
	For background information see the 'Reduced Instruction Set' section
	in `TECHNICAL_DETAILS.md`
	"""
	
	# `else_fn` and `end_fn` are needed outside of RIS_opcodes, so
	# best to use named functions
	def else_fn(code):
		return code.add_op(0, 0, b'\x05', 'else')
	
	def end_fn(code):
		return code.add_op(0, 0, b'\x0b', 'end')
	
	"""
	In terms of stack operands (pop-push):
	
	  sources +1: local.get i32.const
	  
	  neutral  0: 2x block 2x loop else 15x end br return call0 local.tee
	              i32.load i32.load8_u i32.eqz i32.clz i32.ctz i32.popcnt
	  
	    sinks -1: 2x if br_if drop local.set i32.eq i32.ne i32.lt_u i32.gt_u
	              i32.le_u i32.ge_u i32.add i32.sub i32.mul i32.div_u i32.rem_u
	              i32.and i32.or i32.xor i32.shl i32.shr_u i32.rotl i32.rotr
	  
	          -2: select i32.store i32.store8
	
	  weighted sum of sources = +2
	  weighted sum of sinks   = -29
	
	So ideally would want each source x14.5 to balance things out, but then the 
	control instructions etc get swamped by random arithmetic ones. So let's 
	keep the sources somewhat lower and also increase the control and memory 
	instructions (except local.get) x2 to compensate a bit. Amping up the 
	control and memory instructions increases the weighted sum of sinks to:
	
	  -1*(2*5 + 1*18) - 2*(2*3) = -40
	 
	so let's pick x16 for the sources so the final balance is +32 vs -40.
	
	""" 
	RIS_opcodes = (
		# operand sources
		*(
			# i32.const
			# XXX TODO: weirdly this only seems to work up to 31 bits?!?
			lambda code: code.add_op(0, 1, b'\x41' + uLEB128(rnd_i32_0s1s())),
			
			# local.get x - HARDCODED 4 LOCALS
			lambda code: code.add_op( 0, 1, 
				b'\x20' + uLEB128(random.randrange(0x04)) ),
		)*16,
		
		# control and memory instructions
		*(
			# block bt=void
			lambda code: code.add_op(0, 0, b'\x02\x40', 'block', targ=0),
			# block bt=int32
			lambda code: code.add_op(0, 0, b'\x02\x7f', 'block', targ=1),
			
			# loop bt=void
			lambda code: code.add_op(0, 0, b'\x03\x40', 'loop', targ=0),
			# loop bt=int32
			lambda code: code.add_op(0, 0, b'\x03\x7f', 'loop', targ=1),
			
			# if bt=void
			lambda code: code.add_op(1, 0, b'\x04\x40', 'if', targ=0),
			# if bt=int32
			lambda code: code.add_op(1, 0, b'\x04\x7f', 'if', targ=1),
			
			# else
			else_fn,
			
			# end
			# should be x6 (or x7?) to balance out block starts with as many
			# ends, but need more ends as fewer of them get accepted
			*( end_fn, )*16,
			
			# br l
			lambda code: code.add_op( 0, 0, b'\x0c', 'br', 
				dest=random.randrange(len(code.SP)) ),
			
			# br_if l
			lambda code: code.add_op( 1, 0, b'\x0d', 'br_if',
				dest=random.randrange(len(code.SP)) ),
			
			# return
			lambda code: code.add_op(0, 0, b'\x0f', 'return'),
			
			
			# call x=0 (i.e. a recursive call to itself)
			lambda code: code.add_op(0, 0, b'\x10\x00'),
			
			
			# drop
			lambda code: code.add_op(1, 0, b'\x1a'),
			
			# select
			lambda code: code.add_op(3, 1, b'\x1b'),
			
			
			# local.set x - HARDCODED 4 LOCALS
			lambda code: code.add_op( 1, 0,
				b'\x21' + uLEB128(random.randrange(0x04))),
			
			# local.tee x - HARDCODED 4 LOCALS
			lambda code: code.add_op( 1, 1,
				b'\x22' + uLEB128(random.randrange(0x04))),
			
			# i32.load - HARDCODED POOR ALIGN (0x00)
			lambda code: code.add_op(1, 1,
				b'\x28\x00' + uLEB128(rnd_i32_0s1s())),
			
			# i32.load8_u - HARDCODED POOR ALIGN (0x00)
			lambda code: code.add_op(1, 1,
				b'\x2d\x00' + uLEB128(rnd_i32_0s1s())),
			
			# i32.store - HARDCODED POOR ALIGN (0x00)
			lambda code: code.add_op(2, 0,
				b'\x36\x00' + uLEB128(rnd_i32_0s1s())),
			
			# i32.store8 - HARDCODED POOR ALIGN (0x00)
			lambda code: code.add_op(2, 0,
				b'\x3a\x00' + uLEB128(rnd_i32_0s1s())),
		)*2,
		
		# i32.eqz
		lambda code: code.add_op(1, 1, b'\x45'),
		
		# i32.clz
		lambda code: code.add_op(1, 1, b'\x67'),
		
		# i32.ctz
		lambda code: code.add_op(1, 1, b'\x68'),
		
		# i32.popcnt
		lambda code: code.add_op(1, 1, b'\x69'),
		
		# i32.eq
		lambda code: code.add_op(2, 1, b'\x46'),
		
		# i32.ne
		lambda code: code.add_op(2, 1, b'\x47'),
		
		# i32.lt_u
		lambda code: code.add_op(2, 1, b'\x49'),
		
		# i32.gt_u
		lambda code: code.add_op(2, 1, b'\x4b'),
		
		# i32.le_u
		lambda code: code.add_op(2, 1, b'\x4d'),
		
		# i32.ge_u
		lambda code: code.add_op(2, 1, b'\x4f'),
		
		# i32.add
		lambda code: code.add_op(2, 1, b'\x6a'),
		
		# i32.sub
		lambda code: code.add_op(2, 1, b'\x6b'),
		
		# i32.mul
		lambda code: code.add_op(2, 1, b'\x6c'),
		
		# i32.div_u
		lambda code: code.add_op(2, 1, b'\x6e'),
		
		# i32.rem_u
		lambda code: code.add_op(2, 1, b'\x70'),
		
		# i32.and
		lambda code: code.add_op(2, 1, b'\x71'),
		
		# i32.or
		lambda code: code.add_op(2, 1, b'\x72'),
		
		# i32.xor
		lambda code: code.add_op(2, 1, b'\x73'),
		
		# i32.shl
		lambda code: code.add_op(2, 1, b'\x74'),
		
		# i32.shr_u
		lambda code: code.add_op(2, 1, b'\x76'),
		
		# i32.rotl
		lambda code: code.add_op(2, 1, b'\x77'),
		
		# i32.rotr
		lambda code: code.add_op(2, 1, b'\x78'),
	)
	
	def __init__(self, n_bytes:int) -> None:
		
		"""
		         SP: "Stack Pointer(s)", sns = types.SimpleNamespace
		        cur: "current" = number of values on the stack right now
		       targ: target number of values on the stack for current block
		             ('end' only emitted when cur=targ)
		    br_targ: target for 'br's branching down to this block
		             (=targ for all blocks *except* loop, where br_targ=0)
		   if_block: is this an 'if' block where an 'else' can be issued?
		  need_else: is this an 'if' block with targ>0 that *needs* an 'else'?
		"""
		self.SP = [ sns(cur=0, targ=0, br_targ=0, 
			if_block=False, need_else=False) ]
		
		# output
		self.b = b''
		
		# is it time to wrap things up?
		# if True, ok to generate the final 'end' instruction for the function
		self.closing_down = False
		
		self.generate(n_bytes)
	
	def generate(self, n_bytes:int) -> None:
		while len(self.b) < n_bytes:
			random.choice(self.RIS_opcodes)(self)
		
		self.closing_down = True
		
		while len(self.SP) > 0:
			random.choice(self.RIS_opcodes)(self)
	
	def _force_block_end(self):
		"""
		*** ONLY CALL THIS AFTER YOU'VE ISSUED A 'BR' OR A 'RETURN' ***
		
		Force a block end. In general ending a block means an 'end', but 
		'if' blocks with targ>0 require an 'else' first, otherwise we'd get
		the following error:
		
		  "else is expected: if block has a type that can't be implemented 
		   with a no-op"
		"""
		cur_SP = self.SP[-1]
		
		# this is the forcing bit: after a 'br' or a 'return', the target is
		# irrelevant as the end won't be reached anyway
		cur_SP.cur = cur_SP.targ
		
		if cur_SP.need_else:
			self.else_fn()
		else:
			self.end_fn()
	
	def add_op(self, pop:int, push:int, b:bytes, 
			op:str=None, targ:int=None, dest:int=None) -> None:
		"""
		Add an instruction to the code
		
		   pop: how many operands (params) are popped from the operand stack
		  push: how many operands (results) are pushed onto the operand stack
		     b: bytes to add to self.b *if* opcode accepted
		    op: operations that require special handling
		  targ: SP[].targ for new blocks (used by 'block', 'loop' and 'if') 
		  dest: destination for branch instructions (used by 'br' and 'br_if')
		"""
		# empty self.SP means we've issued the final 'end' instruction,
		# cannot add any further instructions after that
		if len(self.SP) == 0:
			return
		
		cur_SP = self.SP[-1]
		#print(f"add_op: level={len(self.SP)-1} cur_SP={cur_SP} "
		#	f"closing_down={self.closing_down} op='{op}'")
		
		# most common case FIRST
		if op is None:
			if cur_SP.cur >= pop:
				self.b += b
				cur_SP.cur += -pop + push
		# 2nd most common case
		elif op == 'end':
			# 1st line: no level 0 'end' until we're closing_down
			if (self.closing_down or len(self.SP) > 1) \
					and cur_SP.cur == cur_SP.targ:
				if cur_SP.need_else:
					# issue an 'else' instead of an 'end'
					self.else_fn()
				else:
					self.b += b
					self.SP.pop()
					
					if len(self.SP) > 0:
						self.SP[-1].cur += cur_SP.targ
		# 3rd most common case
		elif op in ('block', 'loop', 'if'):
			#                        no new blocks once closing_down
			if cur_SP.cur >= pop and not self.closing_down:
				self.add_op(pop, push, b)
				self.SP.append(
					sns( cur=0, targ=targ,
						br_targ=0 if op=='loop' else targ, 
						if_block=(op=='if'),
						need_else=True if op=='if' and targ>0 else False ) )
		elif op == 'else':
			if cur_SP.if_block is True and cur_SP.cur == cur_SP.targ:
				self.b += b
				cur_SP.cur = 0
				cur_SP.if_block = False
				cur_SP.need_else = False
		elif op == 'return':
			# 1st line: we'll probably issue an 'end' as well, but 
			#           "no level 0 'end' until we're closing_down"
			# 2nd line: will be needed if/when the function has br_targ>0
			if (self.closing_down or len(self.SP) > 1) \
					and cur_SP.cur >= self.SP[0].br_targ:
				self.b += b
				self._force_block_end()
		elif op == 'br':
			if cur_SP.cur >= self.SP[-1-dest].br_targ \
					and (self.closing_down or len(self.SP) > 1):
				self.b += b + uLEB128(dest)
				self._force_block_end()
		elif op == 'br_if':
			if cur_SP.cur-pop >= self.SP[-1-dest].br_targ:
				self.b += b + uLEB128(dest)
				cur_SP.cur -= pop
		else:
			raise ValueError(f"Unknown opcode '{op}'")


class Codeling:
	def __init__(self, json_fname:str=None, wasm_fname:str=None, 
			json:dict=None, gen0=None, deferred=False):
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
		
		# NB read_wasm() is an important part of scoring (validation step)
		# and takes place at that point
		self._init_wasm_fname = wasm_fname
		
		self._json_fname = None
		self._wasm_fname = None
		
		if json_fname is not None:
			if json_fname.endswith('.json'):
				self.read_json(json_fname)
			else:
				raise RuntimeError("'json_fname' needs to end in '.json'")
		elif json is not None:
			self.json = json
		elif gen0 is not None:
			self.gen0(*gen0)
		else:
			raise RuntimeError("need one of ('json_fname', 'json', 'gen0'")
	
	def _deferred_init(self) -> None:
		if self._deferred:
			self.__init__(**self._deferred)
			self._deferred = None
	
	def gen(self) -> int:
		return int( re.match('^([0-9a-f]{4})-', self.json['ID']).group(1), 16 )
	
	def b(self) -> bytes:
		return bytes.fromhex(self.json['code'])
	
	def read_json(self, fname:str):
		with open(fname, 'r') as f:
			self.json = json.loads(f.read())
		
		if 'json_version' not in self.json:
			self.json['json_version'] = 1
		
		self._json_fname = fname
	
	def write_json(self, outdir:str) -> str:
		if 'json_version' not in self.json:
			self.json['json_version'] = 1
		
		fname = os.path.join(outdir, self.json['ID'] + '.json')
		with open(fname, 'w') as outf:
			outf.write( json.dumps(self.json, indent=4) )
		
		self._json_fname = fname
		return fname
	
	def read_wasm(self, fname:str):
		config = Config()
		config.consume_fuel = True
		self.store = Store(Engine(config))
		module = Module.from_file(self.store.engine, fname)
		self._instance = Instance(self.store, module, [])
		
		self.mem = self._instance.exports['m']
		assert type(self.mem) is Memory
		
		self._f = self._instance.exports['f']
		assert type(self._f) is Func
		
		self.rnd_addr, self.tmp_addr, self.inp_addr, self.out_addr = \
			[ i32_load(self.mem, addr) for addr in range(0x00, 0x10, 0x04) ]
		
		self._wasm_fname = fname
	
	def write_wasm(self, outdir:str) -> str:
		fname = os.path.join(outdir, self.json['ID'] + '.wasm')
		if os.path.exists(fname):
			return fname
		
		with open(cfg.template, "rb") as f:
			template = f.read()
		
		"""
		XXX TODO UPDATE THIS
		
		See the long comment at the beginning of this file for an overview
		(section "Analysis of template.wasm")
		
		The initial bit is:
		
		   0x01 (1 function)
		
		The initial bit in the inner size() means:
		
		   0x01 (1 locals entry: ) 0x10 (declare 16 locals of ) 0x7f (type i32)
		"""
		wasm = template[:0x23] \
			+ size( b'\x01' + size(b'\x01\x10\x7f' + self.b()) ) \
			+ template[0x36:]
		
		with open(fname, "wb") as f:
			f.write(wasm)
		
		self._wasm_fname = fname
		return fname
	
	def link_json_wasm(self, outdir) -> None:
		for f in (self._json_fname, self._wasm_fname):
			os.link(f, os.path.join(outdir, os.path.basename(f)))
	
	def instance_info(self) -> None:
		d = self._instance.exports._extern_map
		mems = [key for key, val in d.items() if type(val) is Memory]
		print(f'memories exported: {mems}')
		
		funs = [key for key, val in d.items() if type(val) is Func]
		print(f'functions exported: {funs}')
		
		print(f'mem: pages=0x{self.mem.size:x} bytes=0x{self.mem.data_len:x}')
		print(f'rnd addr=0x{self.rnd_addr:06x}')
		print()
	
	def memdump(self) -> None:
		print('dumping memory')
		print('- hdr @ 0x0000:')
		hexdump(bytearray( self.mem.data_ptr[:0x10] ))
		
		for seg, length in \
				( ('rnd', 0x10), ('tmp', 0x40), ('inp', 0x40), ('out', 0x40)):
			addr = getattr(self, seg + '_addr') 
			start = addr & 0xfff0
			print(f'- {seg} @ 0x{addr:04x}:')
			hexdump(bytearray( self.mem.data_ptr[ start : start+length ] ))
			print('   :')
		
		print()
	
	def set_rnd(self) -> None:
		i32_store( self.mem, self.rnd_addr, random.getrandbits(32) )
	
	def set_inp(self, b:bytes) -> None:
		bytes_store( self.mem, self.inp_addr, b )
	
	def get_out(self) -> bytes:
		zero_len = 0
		for i in range(self.out_addr, self.mem.data_len):
			if self.mem.data_ptr[i] == 0x00:
				zero_len += 1
				if zero_len == EOF_LEN:
					return bytes(
						self.mem.data_ptr[ self.out_addr : i-EOF_LEN + 1 ] )
			else:
				zero_len = 0
		
		return bytes( self.mem.data_ptr[ self.out_addr : self.mem.data_len ] )
	
	def gen0(self, ID:str, n_bytes:int) -> None:
		while True:
			code = Code(n_bytes)
			if len(code.b) <= 4*n_bytes:
				break
		
		self.json = {
			'ID': ID,
			'code': code.b.hex(),
			'created': nice_now_UTC(),
			'parents': [],
			'created_by': cfg.this_script_release + ' Codeling.gen0()' }
	
	def concat(self, cdl:'Codeling', child_ID:str) -> 'Codeling':
		child = { 
			'ID': child_ID,
			'code': re.sub('0b$', '', self.json['code']) + cdl.json['code'],
			'created': nice_now_UTC(),
			'parents': [ self.json['ID'], cdl.json['ID'] ],
			'created_by': cfg.this_script_release + ' Codeling.concat()' }
		
		return Codeling(json=child)
	
	def run_wasm(self) -> (str, int, float):
		t_run = time.time()
		self.store.add_fuel(50)
		e = None
		
		try:
			self._f()  # function 'f' exported by the WebAssembly module
		except Exception as err:
			e = comment(str(err))
		
		return (e, self.store.fuel_consumed(), time.time() - t_run)
	
	def score_v00(self) -> 'SimpleNamespace':
		"""Checks whether there was any output to memory at all. WARNING: This 
		function is SLOW."""
		
		mem = self.mem.data_ptr[:self.mem.data_len]
		e, fuel, t_run = self.run_wasm()
		
		score, desc = 0x00, 'OK'
		if e:
			score, desc = -0x40, 'runtime exception\n' + e
		elif mem == self.mem.data_ptr[:self.mem.data_len]:
			score, desc = -0x20, 'no output at all'
		
		return sns(ID=self.json['ID'], score=score, desc=desc, fuel=fuel,
			t_run=t_run)
	
	def score_v01(self) -> 'SimpleNamespace':
		"""Checks that the header is intact and whether there was any output to 
		'out'."""
		
		mem = self.mem.data_ptr[:self.rnd_addr]
		e, fuel, t_run = self.run_wasm()
		
		if e:
			score, desc = -0x40, 'runtime exception\n' + e
		elif mem[:self.rnd_addr] != self.mem.data_ptr[:self.rnd_addr]:
			score, desc = -0x30, 'overwrote header'
		else:
			out = self.get_out()
			
			if len(out) == 0:
				score, desc =-0x20, 'no output to out'
			else:
				writes = [ i + self.out_addr for i, b in enumerate(out) if b ]
				score = 0x30 * len(writes) - int( len(self.json['code']) / 2 )
				desc = 'OK - ' + ' '.join( [f'{w:4x}' for w in writes] )
		
		return sns(ID=self.json['ID'], score=score, desc=desc, fuel=fuel,
			t_run=t_run)
		
	def score_v02(self) -> 'SimpleNamespace':
		"""Penalties: -0x01 for every byte of code length, -0x10 for 
		overwriting the header or a runtime exception (incl running out of 
		fuel), -0x40 when no output to 'out'. Reward: +0x40 for every byte 
		written to 'out'."""
		
		mem = self.mem.data_ptr[:self.rnd_addr]
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
		
		if mem[:self.rnd_addr] != self.mem.data_ptr[:self.rnd_addr]:
				score -= 0x10
				desc += ', overwrote header'
		
		out = self.get_out()
		if len(out) == 0:
			score -= 0x40
			desc += ', no output to out'
		else:
			writes = [ i + self.out_addr for i, b in enumerate(out) if b ]
			score += 0x40 * len(writes)
			desc +=  ' - ' + ' '.join( [f'{w:4x}' for w in writes] )
		
		score -= round( len(self.json['code']) / 2 )
		return sns(ID=self.json['ID'], score=score, desc=desc+msg, fuel=fuel,
			t_run=t_run)
	
	def score(self) -> 'SimpleNamespace':
		# generation
		t_gen = time.time()
		self._deferred_init()
		
		wasm_fname = self._init_wasm_fname
		if wasm_fname is None:
			wasm_fname = self.write_wasm(cfg.tmpdir)
			tmp_wasm = True
		else:
			tmp_wasm = False
		
		# validation
		t_valid = time.time()
		try:
			self.read_wasm(wasm_fname)
		except WasmtimeError as e:
			t_score = time.time()
			res = sns( ID=self.json['ID'], score=-0x80, 
				desc='VALIDATION ERROR\n' + comment(str(e)), 
				t_run=0.0 )
		else:
			# scoring (includes running)
			t_score = time.time()
			score_fn = getattr(self, 'score_' + cfg.fn)  # one of score_vXX()
			self.set_rnd()
			self.set_inp( self.b() )
			res = score_fn()
		
		if cfg.thresh is not None and res.score >= cfg.thresh:
			res.status = 'accept'
			
			if tmp_wasm:
				# NB cannot link because /tmp could be on a different device
				shutil.move(self._wasm_fname, cfg.outdir)
			else:
				link2dir(self._wasm_fname, cfg.outdir)
			
			if self._json_fname is None:
				self.write_json(cfg.outdir)
			else:
				link2dir(self._json_fname, cfg.outdir)
		else:
			res.status = 'reject'
			
			if tmp_wasm:
				os.remove(wasm_fname)
		
		res.t_gen = t_valid - t_gen
		res.t_valid = t_score - t_valid
		res.t_score = (time.time() - t_score) - res.t_run
		return res

def score_Codeling(cdl:'Codeling'):
	return cdl.score()

def score_Codelings(cdl_gtor) -> None:
	IDlen = len(cfg.runid)+13 if cfg.runid is not None else 20
	times = 't_gen t_valid t_run t_score'.split()
	t_start = time.time()
	
	print( '#', ' '.join(sys.argv) )
	print( '# Release ' + cfg.release, 'Codeling.score_' + cfg.fn + '()', 
		f"thresh={cfg.thresh:#x}" if cfg.thresh is not None else 'no threshold',
		sep=', ' )
	print( '# Started:', nice_now() )
	print( '#' )
	print( "# All times below are in microseconds, the scores are in hex" )
	print( f"{'# ID':{IDlen}}", *[ f"{s:>7}" for s in times + \
		'fuel score status'.split() + ['n_acc '] ], 'description', sep="\t" )
	
	n_scored, n_accepted = 0, 0
	n_scored_prev, t_prev = 0, t_start
	with multiprocessing.Pool(processes=cfg.nproc) as p:
		# `.imap` because `.map` converts the iterable to a list
		# `_unordered` because don't care about order, really
		# *** when debugging use `map` instead for cleaner error messages ***
		# for r in map(score_Codeling, cdl_gtor()):
		for r in p.imap_unordered(score_Codeling, cdl_gtor(), chunksize=20):
			n_scored += 1
			if r.status == 'accept':
				n_accepted += 1
			
			micros = [ round(1e6*getattr(r, t)) for t in times ]
			print( f"{r.ID:{IDlen}}", *[ f"{s:>7}" for s in micros + \
				[r.fuel, f"{r.score:x}", r.status, f"{n_accepted} "] ], r.desc,
				sep="\t" )
			
			if n_scored % 1_000_000 == 0:
				if n_scored == 1_000_000:
					print( f"{'time':23}", 
						*[ f"{s:>15}" for s in ('n_scored/1e6', 'n_accepted',
							'scored/hour') ], sep="\t", file=sys.stderr )
				
				t_now = time.time()
				thrpt = (n_scored - n_scored_prev) / (t_now - t_prev) * 3600
				n_scored_prev, t_prev = n_scored, t_now
				print( nice_now(), 
					*[ f"{i:>15d}" for i in (round(n_scored/1e6), n_accepted) ],
					f"{thrpt:>15.2e}", sep="\t", file=sys.stderr )
	
	print( '# Finished:', nice_now() )
	print( f"# Throughput: {n_scored/(time.time()-t_start)*3600:.2e} " 
		f"scored/hour" )

def gtor_alive() -> 'Codeling':
	for json_fname in all_json_fnames(cfg.indir):
		if STOPPING:
			print(' Caught SIGINT, stopping. Waiting for jobs to finish.',
				file=sys.stderr)
			break
		
		yield Codeling( json_fname=json_fname, wasm_fname=json2wasm(json_fname),
			deferred=True )

def gtor_gen0(N:int) -> 'Codeling':
	for i in range(N):
		if STOPPING:
			print(' Caught SIGINT, stopping. Waiting for jobs to finish.',
				file=sys.stderr)
			break
		
		ID = f"{cfg.runid}-{i:012}"
		yield Codeling(gen0=(ID, cfg.length), deferred=True)

# TODO XXX uses LastID, needs a rewrite
def gtor_concat(gen: int) -> 'Codeling':
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

def hack():
	None

def list_available_fns():
	print("Available scoring functions:")
	
	fns = []
	for fn in dir(Codeling):
		m = re.match("score_(.*)", fn)
		if m:
			code = m.group(1)
			doc = getattr( getattr(Codeling, fn), '__doc__' )
			doc = ' '.join( doc.split() )  # get rid of multiple spaces and \n
			fns.append( (code, doc) )
	
	maxL = max([ len(code) for code, _ in fns ])
	doc_w = shutil.get_terminal_size().columns - maxL - 6
	
	for code, doc in fns:
		wrapped = textwrap.wrap(doc, width=doc_w)
		for c, d in zip([code] + ['']*len(wrapped), wrapped):
			print(f"  {c:{maxL}}  {d}")

def SIGINT_handler(sig, frame):
	global STOPPING  # signal handler, so needs to be explicitly declared
	STOPPING = True

def main():
	"""Generate, mutate and score codelings"""
	
	cmd = sys.argv[0]
	epilogue = f"""\
		Most of the options can also be set in `Config.py`.
		
		Examples:
		  # print out a list of scoring functions and their descriptions
		  {cmd} -fn list
		
		  # generate 10 new generation 0 codelings of default length and print
		  # out their scores
		  {cmd} -gen0 10 run01
		
		  # generate 10 strings of random bytes of length 5, score them with
		  # scoring function 'Codelings.score_v02()', print out their scores
		  # and save those with score >= 0x00 to '{cfg.outdir}'
		  {cmd} -rnd0 10 -length 5 -fn v02 -thresh 0x00 run02
		
		Feel free to use Ctrl-C to gracefully end the script at any point.
		"""
	
	defaults = (
		('length', 7),
		('fuel', 50),
		('fn', 'v02'),
		('thresh', None),
		('nproc', multiprocessing.cpu_count()),
		('template', 'template.wasm') )
	
	# if the user has already set the option in Config.py, keep their value
	for attr, val in defaults:
		if not hasattr(cfg, attr):
			setattr(cfg, attr, val)
	
	default_T = 'no codelings are saved' if cfg.thresh is None else cfg.thresh
	
	cfg.release = get_release()
	cfg.this_script = os.path.basename(__file__)
	cfg.this_script_release =  f"{cfg.this_script} v{cfg.release}"
	
	# used internally for hacking on new features
	if len(sys.argv) > 1 and sys.argv[1] == '-hack':
		hack()
		return
	
	# hard-coded to pre-empt checking for required options
	for i in range(2, len(sys.argv)):
		if sys.argv[i-1] == '-fn' and sys.argv[i] == 'list':
			list_available_fns()
			return
	
	def type_int_ish(s:str):
		try:
			return int(s, 0)
		except ValueError:
			try:
				return round( float(s) )
			except ValueError:
				raise argparse.ArgumentTypeError(
					f"failed to convert '{s}' to an int, try something like "
					"'123', '12e3' or '0xf0'." )
	
	def type_score_fn(s:str):
		fn = 'score_' + s
		if hasattr(Codeling, fn):
			return s
		else:
			raise argparse.ArgumentTypeError(f"there is no '{fn}' in Codeling")
	
	def type_dir_str(s:str):
		if os.path.isdir(s):
			return s
		else:
			raise argparse.ArgumentTypeError(f"'{s}' is not a directory")
	
	parser = argparse.ArgumentParser(
		description=main.__doc__, epilog=textwrap.dedent(epilogue),
		formatter_class=argparse.RawDescriptionHelpFormatter )
	
	cmds = parser.add_mutually_exclusive_group(required=True)
	cmds.add_argument('-alive', action='store_true', help=f"""Score all 
		codelings in '{cfg.indir}'.""")
	cmds.add_argument('-rnd0', type=type_int_ish, metavar='N', help="""Generate
		N new random generation 0 codelings that are completely random strings 
		of bytes. Most codelings generated in this way will not pass the 
		WebAssembly validation step (syntax check).""")
	cmds.add_argument('-gen0', type=type_int_ish, metavar='N', help="""Generate
		N new random generation 0 codelings using the Reduced Instruction Set 
		(see 'progress_so_far.md' for details). All codelings generated in this 
		way will pass the WebAssembly validation step (syntax check).""")
	cmds.add_argument('-mutate', type=type_int_ish, metavar='N', 
		help=f"""Generate N new codelings by mutating the codelings in 
		'{cfg.indir}'.""")
	cmds.add_argument('-concat', type=type_int_ish, metavar='gen', help=f"""For
		all codelings X and Y in '{cfg.indir}' such that at least one is of 
		generation 'gen' (eg '0' or '0x00'), create a new codeling X+Y by 
		concatenating their codes.""")
	
	parser.add_argument('-length', type=type_int_ish, metavar='L', help=f"""For
		options that generate new codelings, set the length (-rnd0) or minimum 
		length (-gen0, -mutate) of new sequences or insertions to L bytes. 
		(Default: {cfg.length})""")
	parser.add_argument('-fuel', type=type_int_ish, metavar='F', help=f"""When 
		running a WebAssembly function, provide it with F units of fuel. This 
		limits the number of instructions that will be executed before the 
		function runs out of fuel, thereby preventing infinite loops. [From 
		`store.rs` in the Wasmtime sources: "Most WebAssembly instructions 
		consume 1 unit of fuel. Some instructions, such as `nop`, `drop`, 
		`block`, and `loop`, consume 0 units, as any execution cost associated 
		with them involves other instructions which do consume fuel."] 
		(Default: {cfg.fuel})""")
	parser.add_argument('-fn', type=type_score_fn, metavar='vXX',
		help=f"""Scoring function to use, e.g. 'v02' for 
		'Codeling.score_v02()'. 'list' lists all scoring functions along with 
		their descriptions and exits. (Default: '{cfg.fn}')""")
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
	parser.add_argument('-tmpdir', type=type_dir_str, metavar='path', 
		help=f"""Change the directory where temporary files are stored to 
		'path'. (Default: '{cfg.tmpdir}')""")
	parser.add_argument('-nproc', type=type_int_ish, metavar='N', help=f"""Set
		the number of worker processes to use in the pool to N. (Default for 
		this machine: {cfg.nproc})""")
	
	parser.add_argument('runid', type=str, nargs='?', help=f"""Identifier for 
		this run (e.g. 'run01'). All codelings produced during the run will 
		have identifiers of the form 'runid-012345678901', i.e. the run 
		identifier followed by a dash and twelve digits (with most of the 
		leading ones being zero). The run identifier is a required argument for 
		all types of runs except '-alive', where no new codelings are 
		produced.""")
	
	args = parser.parse_args()
	
	new_cdls = (args.rnd0, args.gen0, args.mutate, args.concat)
	if any(new_cdls) is not None and args.runid is None:
		parser.error("'runid' is required for all runs except '-alive'") 
	
	if args.runid is not None and re.match(r'^#', args.runid):
		parser.error("'runid' cannot start with '#'")
	
	for param in 'length fuel fn thresh indir outdir nproc runid'.split():
		a = getattr(args, param)
		if a is not None:
			setattr(cfg, param, a)
	
	if not os.path.exists(cfg.tmpdir):
		os.makedirs(cfg.tmpdir)
	
	gtor = None
	
	if args.alive:
		gtor = gtor_alive
	elif args.rnd0 is not None:
		gtor = None
		sys.exit("SORRY, -rnd0 not implemented yet (well, re-implemented) :-(")
	elif args.gen0 is not None:
		gtor = lambda: gtor_gen0(args.gen0)
	elif args.mutate is not None:
		gtor = None
		sys.exit("SORRY, -mutate not implemented yet :-(")
	elif args.concat is not None:
		gtor = lambda: gtor_concat(cfg.indir, args.concat)
	
	signal.signal(signal.SIGINT, SIGINT_handler)
	score_Codelings(gtor)

if __name__ == "__main__":
	main()
