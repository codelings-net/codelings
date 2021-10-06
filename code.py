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
    
    NOTE: Inverse transform sampling might be quicker for the 1st step (inner 
    call to random.choices()), ought to implement both and compare speeds, see 
    the 2nd example in:
    
    https://en.wikipedia.org/wiki/Inverse_transform_sampling#Examples
    """
    return random.getrandbits(random.choices(_a, weights=_w, k=1)[0])


class Instr:
    """
    A single instruction within a WebAssembly program
    
    Used directly for simple instructions without an immediate, and as a base 
    class for more complicated instructions
    
    Instruction objects are initialised in two steps:
    
    1. A list of all available instructions is created as the class variable
       Function.RIS_source, but this leaves stack_after set to None
    
    2. When append_to() is called on these Instr objects, it creates a copy of 
       itself, completes the initialisation *of the copy* and then appends
       it to the current CodeBlock
    
    Instance variables:
    
      opcode: bytes         binary opcode
      pop: int              number of params popped from the operand stack
      push: int             number of results pushed onto the operand stack
      stack_after: int      number of operands on the stack after this instr
    
    """
    def __init__(opcode: bytes, pop: int, push: int):
        self.opcode = opcode
        self.pop = pop
        self.push = push
        self.stack_after = None
    
    def to_bytes(self):
        return self.opcode
    
    def _append_OK(self, f: 'Function') -> bool:
        if f.last_instr.stack_after < self.pop:
            if not f.cur_blk.any_OK:
                return False
        
        return True
    
    def _update_stack(self, f: 'Function'):
        self.stack_after = f.last_instr.stack_after - self.pop + self.push
    
    def _append(self, f: 'Function'):
        f.append_instr(self)
    
    def append_to(self, f: 'Function') -> bool:
        """returns True if an instruction was appended, False otherwise"""
        
        if not self._append_OK(f):
            return False
        
        cp = copy(self)
        cp._update_stack(f)
        cp._append(f)
        return True


class InstrWithImm(Instr):
    """
    Base class for instructions with immediates (bytes that follow the opcode)
    
    NB imm is initialised in Step 2 (see the docstring for Instr on how
    instruction objects are initialised)
    
    Instance variables:
    
      imm: bytes            binary immediate(s)
    
    """
    def __init__(opcode: bytes, pop: int, push: int):
        super().__init__(opcode, pop, push)
        self.imm = None
    
    def to_bytes(self):
        return self.opcode + self.imm
    
    def _get_imm(self, f: 'Function') -> bool:
        """
        returns True if imm will pass validation, False otherwise
        
        *** TO BE OVERRIDDEN IN DERIVED CLASSES ***
        
        """
        raise NotImplementedError
        
    def append_to(self, f: 'Function') -> bool:
        """returns True if an instruction was appended, False otherwise"""
        
        if not self._append_OK(f):
            return False
        
        cp = copy(self)
        if not cp._get_imm(f):
            del cp
            return False
        
        cp._update_stack(f)
        cp._append(f)
        return True


class FnStart(Instr):
    """
    Dummy intruction inserted at the start of each function
    
    Instr.append_to() assumes that there is a last_instr. All blocks begin with 
    the block instruction (`block` | `loop` | `if`), but that leaves the 
    function-level block, which is where we insert this dummy.
    """ 
    def __init__(self, targ: int):
        super().__init__(opcode=b'', pop=0, push=0)
        self.stack_after = 0


class BlockInstr(InstrWithImm):
    """
    Block instructions: `block`, `loop` and `if`
    
    Instance variables:
    
      block_type: str       'block' | 'loop' | 'if'
      return_type: str      'empty' | 'i32'
    
    """
    def __init__(self, opcode: bytes, pop: int, push: int, block_type: str):
        super().__init__(opcode, pop, push)
        assert block_type in ('block', 'loop', 'if')
        self.block_type = block_type
        self.return_type = None
    
    def _append_OK(self, f: 'Function') -> bool:
        if not f.new_blks_OK:
            return False
        
        if f.last_instr.stack_after < self.pop:
            if not f.cur_blk.any_OK:
                return False
        
        return True
    
    def _update_stack(self, f: 'Function'):
        self.stack_after = 0
    
    def _get_imm(self, f: 'Function') -> bool:
        self.imm = random.choice((b'\x40', b'\x7f'))
        self.return_type = 'empty' if cp.imm == b'\x40' else 'i32'
        return True
    
    def _append(self, f: 'Function'):
        f.new_blk(instr0=self)


class ElseInstr(Instr):
    """The `else` instruction"""
    
    def _append_OK(self, f: 'Function') -> bool:
        if not (f.cur_blk.if_block and f.cur_blk.else_index is None):
            return False
        
        if f.last_instr.stack_after != f.cur_blk.targ:
            if not f.cur_blk.any_OK:
                return False
        
        return True
    
    def _update_stack(self, f: 'Function'):
        self.stack_after = 0
    
    def _append(self, f: 'Function'):
        super()._append(f)
        f.cur_blk.need_else = False
        f.cur_blk.else_index = len(f.cur_blk.content) - 1


class EndInstr(Instr):
    """The `end` instruction"""
    
    def _append_OK(self, f: 'Function') -> bool:
        if f.last_instr.stack_after != f.cur_blk.targ:
            if not f.cur_blk.any_OK:
                return False
        
        if f.cur_blk.L0 and not f.L0_end_OK:
            return False
        
        # `if` blocks with targ>0 need an `else`, otherwise we'd get an error
        if f.cur_blk.need_else:
            if f.creative_OK:
                return Function.else_instr._append_OK(f)
            else:
                return False
        
        return True
    
    def _append(self, f: 'Function'):
        f.end_blk(self)
    
    def append_to(self, f: 'Function') -> bool:
        """returns True if an instruction was appended, False otherwise"""
        
        if not self._append_OK(f):
            return False
        
        if f.cur_blk.need_else and f.creative_OK:
            return Function.else_instr.append_to(f)
        
        cp = copy(self)
        cp._update_stack(f)
        cp._append(f)
        return True


class BranchInstr(InstrWithImm):
    """
    The `br` and `br_if` instructions
    
    Instance variable:
    
      branch_type: str      'br' | 'br_if'
      dest: int             destination block to which the code will branch
                              (0 = current block, 1 = the block one down, etc)
    """
    def __init__(opcode: bytes, pop: int, push: int, branch_type: str):
        super().__init__(opcode, pop, push)
        assert branch_type in ('br', 'br_if')
        self.branch_type = branch_type
        self.dest = None
    
    def _append_OK(self, f: 'Function') -> bool:
        if self.branch_type == 'br':
            if f.cur_blk.L0 and not f.L0_end_OK:
                return False
        
        return True
    
    def _get_imm(self, f: 'Function') -> bool:
        dest = random.randrange(len(f.cur_blk.blocks))
        br_targ = f.cur_blk.blocks[-1 - dest].br_targ
        if f.last_instr.stack_after - self.pop < br_targ:
            if not f.cur_blk.any_OK:
                return False
        
        self.dest = dest
        self.imm = uLEB128(dest)
        return True
    
    def _append(self, f: 'Function'):
        super()._append(f)
        
        if self.branch_type == 'br':
            f.cur_blk.any_OK = True
            
            if f.creative_OK:
                Function.end_instr.append_to(f)   # add an `end` instruction


class ReturnInstr(Instr):
    """The `return` instruction"""
    
    def _append_OK(self, f: 'Function') -> bool:
        if f.last_instr.stack_after < f.cur_blk.blocks[0].targ:
            if not f.cur_blk.any_OK:
                return False
        
        if f.cur_blk.L0 and not f.L0_end_OK:
            return False
        
        return True

    def _append(self, f: 'Function'):
        super()._append(f)
        f.cur_blk.any_OK = True
        
        if f.creative_OK:
            Function.end_instr.append_to(f)       # add an `end` instruction


class CallInstr(InstrWithImm):
    """The `call` instruction"""
    
    def __init__(opcode: bytes):
        """
        pop and push depend on what function is called, so only initialised
        in Step 2 (see the docstring for Instr for details)
        """
        super().__init__(opcode, None, None)
    
    def _append_OK(self, f: 'Function') -> bool:
        """
        stack size can only be tested once the signature of the function
        to be called is known, i.e. once imm is known
        """
        return True
    
    def _get_imm(self, f: 'Function') -> bool:
        # XXX TODO
        # we are assuming only one function with zero params and zero results
        # and are hard-coding a call to this function
        # i.e. a recursive call to itself
        #
        # zero params = always passes validation
        self.pop = 0
        self.push = 0
        self.imm = b'\x00'
        return True


class MemInstr(InstrWithImm):
    """
    Memory instructions:
        `i32.load`, `i32.load8_u`, `i32.store`, `i32.store8`
    """
    def _get_imm(self, f: 'Function') -> bool:
        # hard-coded poor align (0x00)
        # mostly small random offset (mostly 0x00 and 0x01)
        self.imm = b'\x00' + uLEB128(rnd_i32_0s1s())
        return True


class VarInstr(InstrWithImm):
    """
    Instructions dealing with local (and eventually global as well) variables: 
        `local.get`, `local.set` and `local.tee`
    """
    
    def _get_imm(self, f: 'Function') -> bool:
        # XXX TODO: hard-coded 4 locals
        self.imm = uLEB128(random.randrange(0x04))
        return True


class ConstInstr(InstrWithImm):
    """The `i32.const` instruction"""
    
    def _get_imm(self, f: 'Function') -> bool:
        # XXX TODO: weirdly this only seems to work up to 31 bits?!?
        self.imm = uLEB128(rnd_i32_0s1s())
        return True


class CodeBlock(Instr):
    """
    A block of WebAssembly code
    
    Behaves like a pseudo-instruction: opcode is None, but the rest of the 
    variables (pop, push, stack_after) are defined and used, as is to_bytes()
    and append_to()
    
    Instance variables:
      targ: int             target number of operands on the stack
                              (`end` only emitted when stack size == targ)
      br_targ: int          target for `br`s branching down to this block
                              (=targ except for `loop`, where br_targ=0)
      if_block: bool        is this an `if` block?
                              (an `else` can only be issued within an if block)
      need_else: bool       is this an `if` block that *needs* an `else`?
                              (`if` blocks with targ>0 must have an `else`)
      else_index: int       index of the `else` instruction within content,
                              None => not issued yet
      content: list         instructions (and other blocks) within this block
      blocks: list          list of blocks inside which this block is nested
                              (needed for `br`; includes self at the end)
      L0: bool              is this the Level 0 (function level) block?
                              (more readable than doing len(blocks)==1)
      any_OK: bool          set after `br`/`return`, "any instruction is OK"
      
    """
    def __init__(self, instr0: 'Instr', parent: 'CodeBlock', targ: int = None):
        """
        Two ways of calling this constructor:
        
        1. Level 0 block (function level):
        
           CodeBlock(instr0=FnStart(), parent=None, targ=fn_targ)
        
        2. all other blocks:
        
           CodeBlock(instr0=BlockInstr(...), parent)
        
        Anything else will result in an error!
        """
        if parent is None:
            # no parent, we're at Level 0 (function level)
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
        self.any_OK = False
     
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
    
      L0_blk: 'CodeBlock'   the Level 0 block (at function level)
      last_instr: 'Instr'   last instruction that was appended
      cur_blk: 'CodeBlock'  block to which the next instruction will be added
                              (None = the function has been completed)
      
      new_blks_OK: bool     OK to start new blocks?
      L0_end_OK: bool       OK to issue the final function `end` (at Level 0)?
      creative_OK: bool     OK to creatively add/substitute instructions?
                              (e.g. `end` -> `else` in an `if` block that needs
                               an `else`, or `end` after a `br`)
    """
    
    # `else` and `end` need to be called directly when creative_OK is set, 
    # so need named variables outside of RIS_source
    else_instr = ElseInstr(opcode=b'\x05', pop=0, push=0)
    end_instr = EndInstr(opcode=b'\x05', pop=0, push=0)
    
    """
    In terms of stack operands (pop - push):
    
      sources +1: local.get  i32.const
      
      neutral  0: block  loop  else  end  br  return  call0  local.tee
                  i32.load  i32.load8_u  i32.eqz  i32.clz  i32.ctz  i32.popcnt
      
        sinks -1: if  br_if  drop  local.set
                  i32.eq  i32.ne  i32.lt_u  i32.gt_u  i32.le_u  i32.ge_u
                  i32.add  i32.sub  i32.mul  i32.div_u  i32.rem_u
                  i32.and  i32.or  i32.xor
                  i32.shl  i32.shr_u  i32.rotl  i32.rotr
      
        sinks -2: select  i32.store  i32.store8
      
      sum of sources =  +2
      sum of sinks   = -28  [= (-1)*(4+6+5+3+4) + (-2)*(3)]
    
    Ideally we'd want each source x14 to balance things out, but then the 
    control instructions etc would get swamped by random arithmetic ones. 
    
    So let's increase the control and memory instructions (except local.get) 
    to compensate a bit, say x4 for block starts and x2 for everything else.
    
    Amping up the control and memory instructions increases the weighted sum of
    sinks to:
    
        sinks -1: 4x if  4x br_if  2x drop  2x local.set
                  i32.eq  i32.ne  i32.lt_u  i32.gt_u  i32.le_u  i32.ge_u
                  i32.add  i32.sub  i32.mul  i32.div_u  i32.rem_u
                  i32.and  i32.or  i32.xor
                  i32.shl  i32.shr_u  i32.rotl  i32.rotr
      
        sinks -2: 2x select  2x i32.store  2x i32.store8
    
      sum of sinks = (-1)*((4+4+2+2)+6+5+3+4) + (-2)*((2+2+2))  =  -42
    
    so let's pick x16 for the sources so the final balance is +32 vs -42.
    """
    # ((weight0, (instr00, instr01, ...)),
    #  (weight1, (instr10, instr11, ...)), ...)
    RIS_source = (
        # operand sources
        16, (
            ConstInstr(b'\x41', 0, 1),              # i32.const
            VarInstr(b'\x20', 0, 1)                 # local.get x
        ),
        
        # block starts
        4, (
            BlockInstr(b'\x02', 0, 0, 'block'),     # block bt
            BlockInstr(b'\x03', 0, 0, 'loop'),      # loop bt
            BlockInstr(b'\x04', 1, 0, 'if')         # if bt
        ),
        
        # end instruction (NB neutral for stack operands)
        # should be x12 to balance out block starts with as many block ends,
        # but need more ends as fewer of them get accepted
        32, (end_instr),
        
        # remainder of control instructions, and memory instructions
        2, (
            else_instr,                             # else
            BranchInstr(b'\x0c', 0, 0, 'br'),       # br l
            BranchInstr(b'\x0d', 1, 0, 'br_if'),    # br_if l
            ReturnInstr(b'\x0f', 0, 0),             # return
            CallInstr(b'\x10', 0, 0),               # call x
            Instr(b'\x1a', 1, 0),                   # drop
            Instr(b'\x1b', 3, 1),                   # select
            VarInstr(b'\x21', 1, 0),                # local.set x
            VarInstr(b'\x22', 1, 1),                # local.tee x
            MemInstr(b'\x28', 1, 1),                # i32.load m
            MemInstr(b'\x2d', 1, 1),                # i32.load8_u m
            MemInstr(b'\x36', 2, 0),                # i32.store m
            MemInstr(b'\x3a', 2, 0),                # i32.store8 m
        ),
        
        # the great unwashed
        1, (
            Instr(b'\x45', 1, 1),                   # i32.eqz
            Instr(b'\x67', 1, 1),                   # i32.clz
            Instr(b'\x68', 1, 1),                   # i32.ctz
            Instr(b'\x69', 1, 1),                   # i32.popcnt
            Instr(b'\x46', 2, 1),                   # i32.eq
            Instr(b'\x47', 2, 1),                   # i32.ne
            Instr(b'\x49', 2, 1),                   # i32.lt_u
            Instr(b'\x4b', 2, 1),                   # i32.gt_u
            Instr(b'\x4d', 2, 1),                   # i32.le_u
            Instr(b'\x4f', 2, 1),                   # i32.ge_u
            Instr(b'\x6a', 2, 1),                   # i32.add
            Instr(b'\x6b', 2, 1),                   # i32.sub
            Instr(b'\x6c', 2, 1),                   # i32.mul
            Instr(b'\x6e', 2, 1),                   # i32.div_u
            Instr(b'\x70', 2, 1),                   # i32.rem_u
            Instr(b'\x71', 2, 1),                   # i32.and
            Instr(b'\x72', 2, 1),                   # i32.or
            Instr(b'\x73', 2, 1),                   # i32.xor
            Instr(b'\x74', 2, 1),                   # i32.shl
            Instr(b'\x76', 2, 1),                   # i32.shr_u
            Instr(b'\x77', 2, 1),                   # i32.rotl
            Instr(b'\x78', 2, 1),                   # i32.rotr
        )
    )
    
    
    def __init__(self, targ: int):
        self.L0_blk = CodeBlock(instr0=FunStart(), parent=None, targ=targ)
        self.last_instr = self.L0_blk.content[-1]
        self.cur_blk = self.L0_blk
        
        self.new_blks_OK = True
        self.L0_end_OK = True
    
    def append_instr(self, instr: 'Instr'):
        self.cur_blk.content.append(instr)
        self.last_instr = instr
    
    def new_blk(self, instr0: 'Instr'):
        new_blk = CodeBlock(instr0=instr0, parent=f.cur_blk)
        self.cur_blk.content.append(new_blk)
        self.cur_blk = new_blk
        self.last_instr = instr0
    
    def end_blk(self, instr: 'Instr'):
        self.cur_blk.content.append(cp)
        
        if self.cur_blk.L0:
            self.last_instr = None
            self.cur_blk = None
        else:
            self.last_instr = self.cur_blk
            self.cur_blk = self.cur_blk.blocks[-2]
    
    def parse(self, code: bytes):
        pass
    
    
    #####################################
    
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
