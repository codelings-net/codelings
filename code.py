from evolver import uLEB128

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


### TODO ###
### no imm in Instr, derive InstrImm that has a uLEB128 immediate + numerical 
### value for it - imm_b, imm_i ... will simplify parsing later on

class Instr:
    """
    A single instruction within a WebAssembly program
    
    Used directly for simple instructions and as a base class for more 
    complicated ones
    
    The object is initialised in two steps:
    
    1. A list of all available instructions is created as the class variable
       Function.RIS_source, but this leaves imm and stack_after set to None
    
    2. When append_to() is called on these Instr objects, it creates a copy of 
       itself, completes the initialisation *of the copy* and then appends
       it to the CodeBlock
    
    Instance variables:
    
      opcode: bytes         binary opcode
      pop: int              number of params popped from the operand stack
      push: int             number of results pushed onto the operand stack
      imm: bytes            binary immediate
      stack_after: int      number of operands on the stack after this instr
    
    """
    def __init__(opcode: bytes, pop: int, push: int):
        self.opcode = opcode
        self.pop = pop
        self.push = push
        
        self.imm = None
        self.stack_after = None

    def to_bytes(self):
        if self.imm is not None:
            return self.opcode + self.imm
        else:
            return self.opcode
    
    def append_to(self, f: 'Function') -> bool:
        """returns True if an instruction was appended, False otherwise"""
        
        if f.last_instr.stack_after < self.pop:
            return False
        
        cp = copy(self)
        cp.stack_after = f.last_instr.stack_after - self.pop + self.push
        f.cur_blk.content.append(cp)
        f.last_instr = cp
        return True


class FnStart(Instr):
    """
    Dummy intruction inserted at the start of each function
    
    Instr.append_to() assumes that there is a last_instr. All blocks begin with 
    the block instruction (`block` | `loop` | `if`), but that leaves the 
    function-level block - which is where we insert this dummy.
    
    """ 
    def __init__(self, targ: int):
        super().__init__(opcode=b'', pop=0, push=0)
        self.stack_after = 0


class BlockInstr(Instr):
    """
    Block instructions: `block`, `loop` and `if`
    
    Instance variables:
    
      block_type: str       'block' | 'loop' | 'if'
      return_type: str      'empty' | 'i32'
    
    """
    def __init__(self, opcode: bytes, pop: int, push: int, block_type: str):
        super().__init__(opcode, pop, push)
        self.block_type = block_type
        self.return_type = None
     
    def append_to(self, f: 'Function') -> bool:
        """returns True if an instruction was appended, False otherwise"""
        
        if not f.new_blks_OK or f.last_instr.stack_after < self.pop:
            return False
        
        cp = copy(self)
        cp.stack_after = 0
        
        # block type: 0x40 = empty, 0x7f = i32
        cp.imm = random.choice((b'\x40', b'\x7f'))
        cp.return_type = 'empty' if cp.imm == b'\x40' else 'i32'
        
        new_blk = CodeBlock(instr0=cp, parent=f.cur_blk)
        f.cur_blk.content.append(new_blk)
        f.cur_blk = new_blk
        f.last_instr = cp
        return True


class ElseInstr(Instr):
    """The `else` instruction"""
   
    def append_to(self, f: 'Function') -> bool:
        """returns True if an instruction was appended, False otherwise"""
        
        if not (f.cur_blk.if_block and f.cur_blk.else_index is None):
            return False
        
        if f.last_instr.stack_after != f.cur_blk.targ:
            return False
        
        cp = copy(self)
        cp.stack_after = 0
        f.cur_blk.need_else = False
        f.cur_blk.else_index = len(f.cur_blk.content)
        f.cur_blk.content.append(cp)
        f.last_instr = cp
        return True


class EndInstr(Instr):
    """The `end` instruction"""
    
    def append_to(self, f: 'Function') -> bool:
        """returns True if an instruction was appended, False otherwise"""
        
        if f.cur_blk.L0 and not f.L0_end_OK:
            return False
        
        if f.last_instr.stack_after != f.cur_blk.targ:
            return False
        
        # `if` blocks with targ>0 need an `else`, otherwise we'd get an error
        if f.cur_blk.need_else:
            if f.creative_OK:
                return Function.else_instr.append_to(f)
            else:
                return False
        
        cp = copy(self)
        cp.stack_after = f.last_instr.stack_after
        f.cur_blk.content.append(cp)
        
        if f.cur_blk.L0:
            f.last_instr = None
            f.cur_blk = None
        else:
            f.last_instr = f.cur_blk
            f.cur_blk = f.cur_blk.blocks[-2]
        
        return True


class BranchInstr(Instr):
    """
    The `br` instruction
    
    Instance variable:
    
      dest: int             destination block to which the code will branch
                              (0 = current block, 1 = the block one down, etc)
    
    """
    def __init__(opcode: bytes, pop: int, push: int):
        super().__init__(opcode, pop, push)
        self.dest = None
    
    def append_to(self, f: 'Function') -> bool:
        """returns True if an instruction was appended, False otherwise"""
        
        if f.cur_blk.L0:
            if f.L0_end_OK:
                self.dest = 0
            else:
                return False
        else:
            self.dest = random.randrange(len(f.cur_blk.blocks))
        
        if f.creative_OK and self.dest == 0:
            return Function.end_instr(f)
        
        if f.last_instr.stack_after < f.cur_blk.blocks[-1 - self.dest].br_targ:
            return False
        
        cp = copy(self)
        cp.imm = uLEB128(self.dest)
        cp.stack_after = f.cur_blk.targ       # preparation for a forced `end`
        
        f.cur_blk.content.append(cp)
        f.last_instr = cp
        
        Function.end_instr.append_to(f)       # force an `end` instruction
        return True


class BranchIfInstr(Instr):
    """
    The `br_if` instruction
    
    Instance variable:
    
      dest: int             destination block to which the code will branch
                              (0 = current block, 1 = the block one down, etc)
    
    """
    def __init__(opcode: bytes, pop: int, push: int):
        super().__init__(opcode, pop, push)
        self.dest = None
    
    def append_to(self, f: 'Function') -> bool:
        """returns True if an instruction was appended, False otherwise"""
        
        self.dest = random.randrange(len(f.cur_blk.blocks))
        br_targ = f.cur_blk.blocks[-1 - self.dest].br_targ
        if f.last_instr.stack_after - self.pop < br_targ:
            return False
        
        cp = copy(self)
        cp.imm = uLEB128(self.dest)
        cp.stack_after = f.last_instr.stack_after - self.pop
        
        f.cur_blk.content.append(cp)
        f.last_instr = cp
        return True


class ReturnInstr(Instr):
    """The `return` instruction"""
    """
            # return
            lambda code: code.add_op(0, 0, b'\x0f', 'return'),
            
        elif op == 'return':
            # 1st line: we'll probably issue an 'end' as well, but
            #           "no level 0 'end' until we're closing_down"
            # 2nd line: will be needed if/when the function has br_targ>0
            if (self.closing_down or len(self.SP) > 1) \
                    and cur_SP.cur >= self.SP[0].br_targ:
                self.b += b
                self._force_block_end()
        
        # NB do this bit after `br` and `return` - flush the stack
        cur_SP.cur = cur_SP.targ
    """
    pass


class CallInstr(Instr):
    """
            # call x=0 (i.e. a recursive call to itself)
            lambda code: code.add_op(0, 0, b'\x10\x00'),
    """
    pass


class MemInstr(Instr):
    """
            # i32.load8_u - HARDCODED POOR ALIGN (0x00)
            lambda code: code.add_op(1, 1,
                b'\x2d\x00' + uLEB128(rnd_i32_0s1s())),
            
            # i32.store - HARDCODED POOR ALIGN (0x00)
            lambda code: code.add_op(2, 0,
                b'\x36\x00' + uLEB128(rnd_i32_0s1s())),
            
            # i32.store8 - HARDCODED POOR ALIGN (0x00)
            lambda code: code.add_op(2, 0,
                b'\x3a\x00' + uLEB128(rnd_i32_0s1s())),
    """
    pass


class VarInstr(Instr):
    """The `local.get`, `local.set` and `local.tee` instructions"""
    """
            # local.get x - HARDCODED 4 LOCALS
            lambda code: code.add_op(0, 1,
                b'\x20' + uLEB128(random.randrange(0x04))),
            
            # local.set x - HARDCODED 4 LOCALS
            lambda code: code.add_op(1, 0,
                b'\x21' + uLEB128(random.randrange(0x04))),
            
            # local.tee x - HARDCODED 4 LOCALS
            lambda code: code.add_op(1, 1,
                b'\x22' + uLEB128(random.randrange(0x04))),
    """
    pass


class ConstInstr(Instr):
    """The `i32.const` instruction"""
    """
            # i32.const
            # XXX TODO: weirdly this only seems to work up to 31 bits?!?
            lambda code: code.add_op(0, 1, b'\x41' + uLEB128(rnd_i32_0s1s())),
    """
    pass


class CodeBlock(Instr):
    """
    A block of WebAssembly code
    
    Behaves like a pseudo-instruction: opcode and imm are None, but the rest
    of the variables (pop, push, stack_after) are defined and used
    
    Instance variables:
      targ: int             target number of operands on the stack
                              (`end` only emitted when stack size == targ)
      br_targ: int          target for `br`s branching down to this block
                              (=targ except for `loop`, where br_targ=0)
      if_block: bool        is this an `if` block?
                              (an `else` can only be issued within an if block)
      need_else: bool       is this an `if` block that needs an `else`?
                              (blocks with targ>0 must have an `else`)
      else_index: int       index of the `else` instruction within content,
                              None => not issued yet
      content: list         instructions (and other blocks) within this block
      blocks: list          list of blocks inside which this block is nested
                              (needed for `br`; includes self at the end)
      L0: bool              is this the Level 0 (=function level) block?
      
    """
    def __init__(self, instr0: 'Instr', parent: 'CodeBlock',
                 targ: int = None):
        """
        Two ways of calling this constructor:
        
        1. Level 0 block:
        
           CodeBlock(instr0=FnStart(), parent=None, targ=fn_targ)
        
        2. all other blocks:
        
           CodeBlock(instr0=BlockInstr(...), parent)
        
        Anything else will result in an error!
        
        """
        if parent is None:
            # no parent, we're at Level 0
            assert type(instr0) is FnStart
            assert targ is not None
            
            super().__init__(opcode=None, pop=0, push=targ)
            
            self.targ = targ
            self.br_targ = targ
            self.if_block = False
            self.need_else = False
            
            self.blocks = [self]
            self.L0 = True
        else:
            # have parent, we're at Level > 0
            assert type(instr0) is BlockInstr
            assert targ is None
            
            assert instr0.return_type in ('empty', 'i32')
            targ = 1 if instr0.return_type == 'i32' else 0
            
            super().__init__(opcode=None, pop=instr0.pop, push=targ)
            
            self.targ = targ
            self.br_targ = 0 if instr0.block_type == 'loop' else targ
            self.if_block = (instr0.block_type == 'if')
            self.need_else = (self.if_block and self.targ > 0)
            
            self.blocks = parent.blocks + [self]
            self.L0 = False
        
        self.else_index = None
        self.content = [instr0]
     
    def to_bytes(self):
        return b''.join([instr.to_bytes() for instr in self.content])


class Function:
    """
    - Generate random functions that pass WebAssembly validation
    
    - Parse existing functions and mutate them
    
    Used by the `-gen0` and `-mutate` options in main() in `evolver.py`
    
    For background information see the 'Reduced Instruction Set' section
    in `TECHNICAL_DETAILS.md` XXX TODO
    
    Instance variables:
    
      L0_blk: 'CodeBlock'   the Level 0 block (a.k.a. root or function level)
      last_instr: 'Instr'   last instruction that was appended
      cur_blk: 'CodeBlock'  block to which the next instruction will be added
                              (None = the function has been completed)
      
      new_blks_OK: bool     OK to start new blocks?
      L0_end_OK: bool       OK to issue the final function `end` (at Level 0)?
      creative_OK: bool     OK to creatively add/substitute instructions?
                              (e.g. `end` -> `else` in an `if` block that needs
                               an `else`, or `end` after a `br`)
    
    """
    
    
    def __init__(self, targ: int):
        self.L0_blk = CodeBlock(instr0=FunStart(), parent=None, targ=targ)
        self.last_instr = self.L0_blk.content[-1]
        self.cur_blk = self.L0_blk
        
        self.new_blks_OK = True
        self.L0_end_OK = True
    
    def parse(self, code: bytes):
        pass
    
    
    #################
    
    # `else` and `end` need to be called directly when creative_OK is set, 
    # so need named variables outside of RIS_xxx
    else_instr = ElseInstr(opcode=b'\x05', pop=0, push=0)
    end_instr = EndInstr(opcode=b'\x05', pop=0, push=0)
    
    # ((weight0, [instr00, instr01, ...]),
    #  (weight1, [instr10, instr11, ...]), ...)
    RIS_source = ( ()
    

    """
    In terms of stack operands (pop - push):
    
      sources +1: local.get i32.const
      
      neutral  0: 2x block  2x loop  else  16x end  br  return  call0
                  local.tee  i32.load  i32.load8_u  i32.eqz  i32.clz  i32.ctz
                  i32.popcnt
      
        sinks -1: 2x if  br_if  drop  local.set  i32.eq  i32.ne  i32.lt_u
                  i32.gt_u  i32.le_u  i32.ge_u  i32.add  i32.sub  i32.mul
                  i32.div_u  i32.rem_u  i32.and  i32.or  i32.xor  i32.shl
                  i32.shr_u  i32.rotl  i32.rotr
      
              -2: select  i32.store  i32.store8
      
      weighted sum of sources =  +2
      weighted sum of sinks   = -29
      
      [where (weighted sum) = (pop - push)*(sum of RIS weights) ]
    
    So ideally would want each source x14.5 to balance things out, but then the
    control instructions etc get swamped by random arithmetic ones. So let's
    keep the sources somewhat lower and also increase the control and memory
    instructions (except local.get) x2 to compensate a bit. Amping up the
    control and memory instructions increases the weighted sum of sinks to:
    
      (-1)*(2*5 + 1*18) + (-2)*(2*3) = -40
    
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
