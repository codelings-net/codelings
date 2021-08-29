from evolver import uLEB128

from dataclasses import dataclass
from copy import copy

import random


# TODO XXX get rid of this
from types import SimpleNamespace as sns


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
    
    Inverse transform sampling might be quicker for the 1st step, ought to
    implement both and compare speeds - see the 2nd example in:
    
    https://en.wikipedia.org/wiki/Inverse_transform_sampling#Examples
    """
    return random.getrandbits(random.choices(_a, weights=_w, k=1)[0])


@dataclass
class Instr:
    """
    A single instruction within a WebAssembly program
    
    Used directly for simple instructions and as a base class for more 
    complicated ones.
    
    The object is initialised in two steps:
    
    1. A list of all available instructions is created as the class variable
       Code.RIS, but this leaves imm, prev_instr and stack_after set to None
    
    2. When append() is called on these Instr objects, it creates a copy of 
       itself, completes the initialisation *of the copy* and then appends
       it to the CodeBlock
    """
    opcode: bytes           # binary opcode
    pop: int                # number of params popped from the operand stack
    push: int               # number of results pushed onto the operand stack
    imm: bytes \            # binary immediate
        = field(default=None, init=False)
    prev_instr: 'Instr' \   # previous instruction within a CodeBlock
        = field(default=None, init=False)
    stack_after: int \      # number of operands on the stack after this instr
        = field(default=None, init=False)
    
    def to_bytes(self):
        if imm is not None:
            return opcode + imm
        else:
            return opcode
    
    def append(self, blk: 'CodeBlock', c: 'Code') -> (bool, 'CodeBlock'):
        prev_instr = blk.content[-1]
        if prev_instr.stack_after < self.pop:
            return (False, blk)
        
        cp = copy(self)
        cp.prev_instr = prev_instr
        cp.stack_after = prev_instr.stack_after - self.pop + self.push
        blk.content.append(cp)
        return (True, blk)


@dataclass
class DummyInstr(Instr):
    """
    Dummy intruction inserted at the start of each function
    
    Instr.append() assumes that there always is a prev_instr. All blocks begin 
    with the block instruction (`block` | `loop` | `if`), but that leaves the 
    function level - which is where we insert this dummy.
    """ 
    def __init__(self):
        self.opcode = b''
        self.pop = 0
        self.push = 0
        self.stack_after = 0


@dataclass
class BlockInstr(Instr):
    """Block instructions: `block`, `loop` and `if`"""
    
    instr_type: str         # 'block' | 'loop' | 'if'
     
    def append(self, blk: 'CodeBlock', c: 'Code') -> (bool, 'CodeBlock'):
        if c.closing_down:
            return (False, blk)
        
        prev_instr = blk.content[-1]
        if prev_instr.stack_after < self.pop:
            return (False, blk)
        
        cp = copy(self)
        cp.stack_after = 0
        
        # block type: 0x40 = empty, 0x7f = i32
        cp.imm = random.choice((b'\x40', b'\x7f'))
        
        targ = 1 if cp.imm == b'\x7f' else 0
        br_targ = 0 if cp.instr_type == 'loop' else targ
        if_block = (cp.instr_type == 'if')
        need_else = (if_block and targ > 0)
        new_blk = CodeBlock(targ, br_targ, if_block, need_else, blk, cp)
        return (True, new_blk)


@dataclass
class ElseInstr(Instr):
    """The `else` instruction"""
   
    def append(self, blk: 'CodeBlock', c: 'Code') -> (bool, 'CodeBlock'):
        if not (blk.if_block and blk.else_index is None):
            return (False, blk)
        
        prev_instr = blk.content[-1]
        if prev_instr.stack_after != blk.targ:
            return (False, blk)
        
        cp = copy(self)
        cp.stack_after = 0
        blk.else_index = len(blk.content)
        blk.content.append(cp)
        return (True, blk)


@dataclass
class EndInstr(Instr):
    """The `end` instruction"""
    
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
    
    def append(self, blk: 'CodeBlock', c: 'Code') -> (bool, 'CodeBlock'):
        if blk.parents is None and not c.closing_down:
            return (False, blk)
        
        if blk.need_else:
            return c.else_instr(blk, c)
        
        prev_instr = blk.content[-1]
        if prev_instr.stack_after != blk.targ:
            return (False, blk)
        
        cp = copy(self)
        cp.stack_after = prev_instr.stack_after
        blk.else_index = len(blk.content)
        blk.content.append(cp)
        return (True, blk)


"""
        else:
            # block type: 0x40 = empty, 0x7f = i32
            self.imm = random.choice((b'\x40', b'\x7f'))
            
            targ = 1 if self.imm == b'\x7f' else 0
            br_targ = 0 if instr_type == 'loop' else targ
            if_block = (instr_type == 'if')
            need_else = (if_block and targ > 0)
            new_blk = CodeBlock(targ, br_targ, if_block, need_else, blk)
            new_blk.content.append(self)
            return (True, new_blk)    

        imm     class of instruction immediate
        spec    code for instructions that require special handling
        targ    stack target for new blocks (used by 'block', 'loop' and 'if')
        dest    destination for branch instructions (used by 'br' and 'br_if')

                 imm: str = None,
                 spec: str = None,
                 targ: int = None,
                 dest: int = None):
"""

@dataclass
class CodeBlock(Instr):
    """
    A block of WebAssembly code
    
    Behaves like a pseudo-instruction: opcode and imm are None, but 
    the rest of the variables are defined and used
    
    """
    
    targ: int                # target number of operands on the stack
                             # (`end` only emitted when stack size == targ)
    br_targ: int             # target for `br`s branching down to this block
                             # (=targ except for `loop`, where br_targ=0)
    if_block: bool           # is this an `if` block?
                             # (an `else` can only be issued within an if block)
    need_else: bool          # is this an `if` block that needs an `else`?
                             # (targ>0 blocks must have an `else` part)
    else_index: int = None   # index of the `else` instruction within content
    content: list = None     # instructions and other blocks inside this block
    parents: list = None     # list of blocks inside which this block is nested
                             # (needed for `br`; None => this block is Level 0)
    stack_after: int = None
    
    def __init__(self, targ: int, br_targ: int, if_block: bool, need_else: bool,
                 parent: 'CodeBlock' = None, first_instr: int = DummyInstr()):
        self.targ = targ
        self.br_targ = br_targ
        self.if_block = if_block
        self.need_else = need_else
        
        if parent is None:
            self.parents = []
        else:
            self.parents = copy(parent.parents)
            self.parents.append(parent)
        
        self.content = [first_instr]
    
    def to_bytes(self):
        return b''.join([i.to_bytes() for i in self.content])
    
    def append(self, blk: 'CodeBlock', c: 'Code') -> (bool, 'CodeBlock'):
        raise 
    


class Code:
    """
    - Generate random codes that pass WebAssembly validation
    
    - Parse existing codes and mutate them
    
    Used by the `-gen0` and `-mutate` options in main() in `evolver.py`
    
    For background information see the 'Reduced Instruction Set' section
    in `TECHNICAL_DETAILS.md` XXX TODO
    """
    
    else_instr = ElseInstr(0, 0, b'\x05')
    end_instr = EndInstr(0, 0, b'\x05')
    RIS = ()
    
    def __init__(self):
        self.closing_down = False
    
    
    
    # `else_fn` and `end_fn` are needed outside of RIS_opcodes,
    # so need named functions
    def else_fn(code):
        return code.add_op(0, 0, b'\x05', 'else')
    
    def end_fn(code):
        return code.add_op(0, 0, b'\x0b', 'end')
    
    """
    In terms of stack operands (pop-push):
    
      sources +1: local.get i32.const
      
      neutral  0: 2x block  2x loop  else  16x end  br  return  call0
                  local.tee  i32.load  i32.load8_u  i32.eqz  i32.clz  i32.ctz
                  i32.popcnt
      
        sinks -1: 2x if  br_if  drop  local.set  i32.eq  i32.ne  i32.lt_u
                  i32.gt_u  i32.le_u  i32.ge_u  i32.add  i32.sub  i32.mul
                  i32.div_u  i32.rem_u  i32.and  i32.or  i32.xor  i32.shl
                  i32.shr_u  i32.rotl  i32.rotr
      
              -2: select  i32.store  i32.store8
      
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
            lambda code: code.add_op(0, 1,
                b'\x20' + uLEB128(random.randrange(0x04))),
        ) * 16,
        
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
            # should be x6 to balance out block starts with as many block ends,
            # but need more ends as fewer of them get accepted
            *(end_fn, ) * 16,
            
            # br l
            lambda code: code.add_op(0, 0, b'\x0c', 'br',
                dest=random.randrange(len(code.SP))),
            
            # br_if l
            lambda code: code.add_op(1, 0, b'\x0d', 'br_if',
                dest=random.randrange(len(code.SP))),
            
            # return
            lambda code: code.add_op(0, 0, b'\x0f', 'return'),
            
            # call x=0 (i.e. a recursive call to itself)
            lambda code: code.add_op(0, 0, b'\x10\x00'),
            
            # drop
            lambda code: code.add_op(1, 0, b'\x1a'),
            
            # select
            lambda code: code.add_op(3, 1, b'\x1b'),
            
            # local.set x - HARDCODED 4 LOCALS
            lambda code: code.add_op(1, 0,
                b'\x21' + uLEB128(random.randrange(0x04))),
            
            # local.tee x - HARDCODED 4 LOCALS
            lambda code: code.add_op(1, 1,
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
        ) * 2,
        
        lambda code: code.add_op(1, 1, b'\x45'),		# i32.eqz
        lambda code: code.add_op(1, 1, b'\x67'),		# i32.clz
        lambda code: code.add_op(1, 1, b'\x68'),		# i32.ctz
        lambda code: code.add_op(1, 1, b'\x69'),		# i32.popcnt
        lambda code: code.add_op(2, 1, b'\x46'),		# i32.eq
        lambda code: code.add_op(2, 1, b'\x47'),		# i32.ne
        lambda code: code.add_op(2, 1, b'\x49'),		# i32.lt_u
        lambda code: code.add_op(2, 1, b'\x4b'),		# i32.gt_u
        lambda code: code.add_op(2, 1, b'\x4d'),		# i32.le_u
        lambda code: code.add_op(2, 1, b'\x4f'),		# i32.ge_u
        lambda code: code.add_op(2, 1, b'\x6a'),		# i32.add
        lambda code: code.add_op(2, 1, b'\x6b'),		# i32.sub
        lambda code: code.add_op(2, 1, b'\x6c'),		# i32.mul
        lambda code: code.add_op(2, 1, b'\x6e'),		# i32.div_u
        lambda code: code.add_op(2, 1, b'\x70'),		# i32.rem_u
        lambda code: code.add_op(2, 1, b'\x71'),		# i32.and
        lambda code: code.add_op(2, 1, b'\x72'),		# i32.or
        lambda code: code.add_op(2, 1, b'\x73'),		# i32.xor
        lambda code: code.add_op(2, 1, b'\x74'),		# i32.shl
        lambda code: code.add_op(2, 1, b'\x76'),		# i32.shr_u
        lambda code: code.add_op(2, 1, b'\x77'),		# i32.rotl
        lambda code: code.add_op(2, 1, b'\x78'),		# i32.rotr
    )
    
    def __init__(self, n_bytes: int) -> None:
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
        self.SP = [sns(cur=0, targ=0, br_targ=0,
                       if_block=False, need_else=False)]
        
        # output
        self.b = b''
        
        # is it time to wrap things up?
        # if True, ok to generate the final 'end' instruction for the function
        self.closing_down = False
        
        self.generate(n_bytes)
    
    def generate(self, n_bytes: int) -> None:
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
    
    def add_op(self, pop: int, push: int, b: bytes,
               op: str = None, targ: int = None, dest: int = None) -> None:
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
        # print(f"add_op: level={len(self.SP)-1} cur_SP={cur_SP} "
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
                    sns(cur=0, targ=targ,
                        br_targ=0 if op == 'loop' else targ,
                        if_block=(op == 'if'),
                        need_else=True if op == 'if' and targ > 0 else False))
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
            if cur_SP.cur >= self.SP[-1 - dest].br_targ \
                    and (self.closing_down or len(self.SP) > 1):
                self.b += b + uLEB128(dest)
                self._force_block_end()
        elif op == 'br_if':
            if cur_SP.cur - pop >= self.SP[-1 - dest].br_targ:
                self.b += b + uLEB128(dest)
                cur_SP.cur -= pop
        else:
            raise ValueError(f"Unknown opcode '{op}'")
