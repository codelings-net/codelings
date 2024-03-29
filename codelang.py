import util

import copy
import random


class Instr:
    """A single instruction within a WebAssembly (or CodeLang) program
    
    Used directly for simple instructions without an immediate and as a base 
    class for more complex instructions
    
    Instruction objects are initialised in two steps:
    
    1. The list of all available instructions is stored in RIS_instrs, but the
       Instr objects in that list have stack_after, fn_i and blk_i set to None
    
    2. When append_to() is called on one of these Instr objects, the object 
       creates a copy of itself, completes the initialisation *of the copy* 
       and then appends it to the current CodeBlock
    
    Instance variables:
    
      mnemonic: str         instruction mnemonic ('i32.const', 'if bt' etc) 
      opcode: bytes         binary opcode
      pop: int              number of params popped from the operand stack
      push: int             number of results pushed onto the operand stack
      stack_after: int      number of operands on the stack after this instr
      fn_i: int             this is instruction number fn_i in the function
      blk_i: int            this is instruction number blk_i in its block
    
    """
    def __init__(self, mnemonic: str, opcode: bytes, pop: int, push: int):
        self.mnemonic = mnemonic
        self.opcode = opcode
        self.pop = pop
        self.push = push
        self.stack_after = None
        self.fn_i = None
        self.blk_i = None
    
    def b(self) -> bytes:
        return self.opcode
    
    def _any_OK(self, f: 'Function') -> bool:
        if f.cur_blk.any_OK_after is None:
            return False
        
        blk_i = self.blk_i
        if blk_i is None: blk_i = f.last_instr.blk_i + 1
        return f.cur_blk.any_OK_after < blk_i
    
    def _append_OK(self, f: 'Function') -> bool:
        if f.wind_down:
            targ = f.cur_blk.temp_targ
            if targ is None: targ = f.cur_blk.targ
            
            diff = f.last_instr.stack_after - targ
            if abs(diff) < abs(diff - self.pop + self.push):
                if random.choice((True, False)):
                    return False
        
        return self.pop <= f.last_instr.stack_after or self._any_OK(f)
    
    def _copy(self) -> 'Instr':
        # needed because the CodeBlock pseudo-instruction returns self
        return copy.copy(self)
    
    def _finish_init(self, f: 'Function') -> bool:
        """returns True if the instruction passes validation, 
        False otherwise"""
        
        prev = f.last_instr
        if self.pop <= prev.stack_after:
            self.stack_after = prev.stack_after - self.pop + self.push
        elif self._any_OK(f):
            self.stack_after = self.push
        else:
            return False
            
        self.fn_i = prev.fn_i + 1
        self.blk_i = prev.blk_i + 1
        return True
    
    def _append(self, f: 'Function'):
        f._append_instr(self)
    
    def append_to(self, f: 'Function') -> bool:
        """returns True if an instruction was appended, False otherwise"""
        
        if not self._append_OK(f):
            return False
        
        cp = self._copy()
        if not cp._finish_init(f):
            del cp
            return False
        
        cp._append(f)
        return True
    
    def desc(self, tokens=False):
        if tokens:
            return [('M', self.mnemonic),
                    ('-', str(self.pop)),
                    ('+', str(self.push)),
                    ('S', f"{self.stack_after:2}")]
        else:
            return (f"{self.fn_i:2},{self.blk_i:2}: "
                    f"{self.mnemonic:20} -{self.pop} +{self.push} "
                    f"[{self.stack_after:2}]")


class InstrWithImm(Instr):
    """Base class for instructions with immediates (=bytes after the opcode)
    
    The immediate `imm` is initialised in Step 2 (see the docstring for Instr 
    on how instruction objects are initialised)
    
    Instance variables:
    
      imm: bytes            binary immediate(s)
    
    """
    def __init__(self, mnemonic: str, opcode: bytes, pop: int, push: int):
        super().__init__(mnemonic, opcode, pop, push)
        self.imm = None
    
    def b(self) -> bytes:
        return self.opcode + self.imm
    
    def _update_imm(self, f: 'Function') -> bool:
        """To be overridden in child classes"""
        raise NotImplementedError

    def _finish_init(self, f: 'Function') -> bool:
        return self._update_imm(f) and super()._finish_init(f)


class DummyFnStart(Instr):
    """Dummy intruction inserted at the start of each function
    
    Instr.append_to() assumes that there is a last_instr. All blocks begin 
    with the block instruction (`block` | `loop` | `if`), but that leaves the 
    function-level block, which is where we insert this dummy.
    """ 
    def __init__(self):
        super().__init__(mnemonic='DummyFnStart', opcode=b'', pop=0, push=0)
        self.stack_after = 0
        self.fn_i = -1
        self.blk_i = 0


class BlockInstr(InstrWithImm):
    """Block instructions: `block`, `loop` and `if`
    
    Instance variable:
    
      return_type: str      'empty' | 'i32'
    
    """
    def __init__(self, mnemonic: str, opcode: bytes, pop: int, push: int):
        super().__init__(mnemonic, opcode, pop, push)
        assert mnemonic in ('block bt', 'loop bt', 'if bt')
        self.return_type = None
    
    def _append_OK(self, f: 'Function') -> bool:
        return super()._append_OK(f) and not f.wind_down
    
    def _update_imm(self, f: 'Function') -> bool:
        if self.return_type is None:
            imm_options = (b'\x40', b'\x7f')
            
            if f.bs is None:
                self.imm = random.choice(imm_options)
            else:
                self.imm = f.bs.next_b()
            
            assert self.imm in imm_options
            self.return_type = 'empty' if self.imm == b'\x40' else 'i32'
        else:
            assert self.return_type in ('empty', 'i32')
            self.imm = b'\x40' if self.return_type == 'empty' else b'\x7f'
        
        return True
    
    def _finish_init(self, f: 'Function') -> bool:
        self._update_imm(f)
        self.stack_after = 0
        self.fn_i = f.last_instr.fn_i + 1
        self.blk_i = 0
        return True
    
    def _append(self, f: 'Function'):
        f._new_blk(instr0=self)
    
    def desc(self, tokens=False):
        if tokens:
            imm = [('=', 'return_type='), ('V', str(self.return_type))]
        else:
            imm = f"   return_type={self.return_type}"
        
        return super().desc(tokens) + imm


class ElseInstr(Instr):
    """The `else` instruction"""
    
    def _append_OK(self, f: 'Function') -> bool:
        if (type(f.cur_blk) is not IfBlock or
            f.cur_blk.else_blk is not None or
            f.cur_blk.restr_end or
            (f.wind_down and not f.cur_blk.need_else)):
            #
            return False
        
        return (f.last_instr.stack_after == f.cur_blk.targ or
                (self._any_OK(f) and
                 f.last_instr.stack_after <= f.cur_blk.targ))
    
    def _finish_init(self, f: 'Function') -> bool:
        self.stack_after = 0
        self.fn_i = f.last_instr.fn_i + 1
        self.blk_i = 0
        return True
    
    def _append(self, f: 'Function'):
        f._new_blk(instr0=self)


class EndInstr(Instr):
    """The `end` instruction"""
    
    def _append_OK(self, f: 'Function') -> bool:
        # `if` blocks with targ>0 need an `else`, otherwise we'd get an error
        if type(f.cur_blk) is IfBlock and f.cur_blk.need_else:
            return f.creative_OK and else_instr._append_OK(f)
        
        if f.cur_blk.restr_end and not f.restr_end_OK:
            return False
        
        return (f.last_instr.stack_after == f.cur_blk.targ or
                (self._any_OK(f) and
                 f.last_instr.stack_after <= f.cur_blk.targ))
    
    def _append(self, f: 'Function'):
        f._end_blk(self)
    
    def append_to(self, f: 'Function') -> bool:
        if (type(f.cur_blk) is IfBlock and
            (f.cur_blk.need_else and f.creative_OK)):
            #
            return else_instr.append_to(f)
        else:
            return super().append_to(f)


class BranchInstr(InstrWithImm):
    """The `br` and `br_if` instructions
    
    Instance variable:
    
      dest: int             destination block to which the code will branch
                              (0 = current block, 1 = the block one down, etc)
    """
    def __init__(self, mnemonic: str, opcode: bytes, pop: int, push: int):
        super().__init__(mnemonic, opcode, pop, push)
        assert mnemonic in ('br l', 'br_if l')
        self.dest = None
    
    def _append_OK(self, f: 'Function') -> bool:
        # the bulk of testing is done in _finish_init_imm() once dest is known
        return (super()._append_OK(f) and
                (self.mnemonic != 'br l' or not f.cur_blk.restr_end or
                 f.restr_end_OK))
    
    def _dest_OK(self, f: 'Function', dest: int) -> bool:
        if len(f.cur_blk.blocks) <= dest:
            return False
        
        br_targ = f.cur_blk.blocks[-1 - dest].br_targ
        return (br_targ <= f.last_instr.stack_after - self.pop or
                self._any_OK(f))
    
    def _update_imm(self, f: 'Function') -> bool:
        if self.dest is None:
            if f.bs is None:
                for i in range(10):
                    self.dest = random.randrange(len(f.cur_blk.blocks))
                    if self._dest_OK(f, self.dest):
                        break
            else:
                self.dest = f.bs.next_uLEB128()
        
        if not self._dest_OK(f, self.dest):
            return False
        
        self.imm = util.uLEB128(self.dest)
        return True
    
    def _finish_init(self, f: 'Function') -> bool:
        self._update_imm(f)
        retval = super()._finish_init(f)
        if self.mnemonic == 'br l':
            self.stack_after = 0
            if f.cur_blk.any_OK_after is None:
                f.cur_blk.any_OK_after = self.blk_i
        
        return retval
    
    def _append(self, f: 'Function'):
        f._append_instr(self)
        if self.mnemonic == 'br l' and f.creative_OK:
            end_instr.append_to(f)       # add an `end` instruction
    
    def desc(self, tokens=False):
        if tokens:
            imm = [('=', 'dest='), ('V', str(self.dest))]
        else:
            imm = f"   dest={self.dest}"
        
        return super().desc(tokens) + imm


class ReturnInstr(Instr):
    """The `return` instruction"""
    
    def _append_OK(self, f: 'Function') -> bool:
        targ = f.cur_blk.blocks[0].targ
        return ((not f.cur_blk.restr_end or f.restr_end_OK) and
                (targ <= f.last_instr.stack_after or self._any_OK(f)))
    
    def _finish_init(self, f: 'Function') -> bool:
        retval = super()._finish_init(f)
        self.stack_after = 0
        if f.cur_blk.any_OK_after is None:
            f.cur_blk.any_OK_after = self.blk_i
        
        return retval
    
    def _append(self, f: 'Function'):
        f._append_instr(self)
        if f.creative_OK:
            end_instr.append_to(f)       # add an `end` instruction


class CallInstr(InstrWithImm):
    """The `call` instruction"""
    
    def __init__(self, mnemonic: str, opcode: bytes):
        """pop and push depend on what function is called, so only initialised
        in Step 2 (see the docstring for Instr for details)
        """
        super().__init__(mnemonic, opcode, None, None)
    
    def _append_OK(self, f: 'Function') -> bool:
        """stack size can only be tested once the signature of the function
        to be called is known, i.e. once imm is known
        """
        return True
    
    def _update_imm(self, f: 'Function') -> bool:
        # XXX TODO
        # we are assuming only one function with zero params and zero results
        # and are hard-coding a call to this function (=function no. 0)
        # i.e. a recursive call to itself
        fnID = b'\x00'
        if self.imm is None:
            if f.bs is None:
                self.imm = fnID
            else:
                self.imm = f.bs.next_b()
        
        if self.imm != fnID:
            return False
        
        # pop=0 means that this always passes validation
        self.pop = 0
        self.push = 0

        return True


class MemInstr(InstrWithImm):
    """Memory instructions:
    
      `i32.load`, `i32.load8_u`, `i32.store`, `i32.store8`
    
    Instance variables:
    
      align: int            the align part of memarg
      offset: int           the offset part of memarg
    
    offset is added to the memory baseAddress (which is popped from the stack)
    and the alignment (align) is a promise to the VM that:
    
      (baseAddr + memarg.offset) mod 2^memarg.align == 0
    
      "Unaligned access violating that property is still allowed and must 
       succeed regardless of the annotation. However, it may be substantially 
       slower on some hardware."
    
    """
    def __init__(self, mnemonic: str, opcode: bytes, pop: int, push: int):
        super().__init__(mnemonic, opcode, pop, push)
        self.align = None
        self.offset = None
    
    def _update_imm(self, f: 'Function') -> bool:
        if self.align is None or self.offset is None:
            if f.bs is None:
                # hard-coded poor align (slow but always works)
                # mostly small random offset (mostly 0x00 and 0x01)
                self.align = 0x00
                self.offset = util.rnd_i32_0s1s()
            else:
                self.align = f.bs.next_uLEB128()
                self.offset = f.bs.next_uLEB128()
        
        self.imm = util.uLEB128(self.align) + util.uLEB128(self.offset)
        return True
    
    def regen_imm(self, f: 'Function') -> bool:
        self.offset = None
        return self._update_imm(f)
    
    def desc(self, tokens=False):
        if tokens:
            imm = [('=', 'align='), ('V', str(self.align)),
                   ('=', 'offset='), ('V', str(self.offset))]
        else:
            imm = f"   align={self.align} offset={self.offset}"
        
        return super().desc(tokens) + imm


class VarInstr(InstrWithImm):
    """Instructions dealing with local (and eventually also global) variables:
    
      `local.get`, `local.set` and `local.tee`
    
    Instance variable:
    
      varID: int            ID of variable to be operated on
    
    """
    def __init__(self, mnemonic: str, opcode: bytes, pop: int, push: int):
        super().__init__(mnemonic, opcode, pop, push)
        self.varID = None
    
    def _update_imm(self, f: 'Function') -> bool:
        # XXX TODO: hard-coded 4 locals
        n_vars = 4
        
        if self.varID is None:
            if f.bs is None:
                self.varID = random.randrange(n_vars)
            else:
                self.varID = f.bs.next_uLEB128()
        
        if n_vars <= self.varID:
            return False
        
        self.imm = util.uLEB128(self.varID)
        return True
    
    def regen_imm(self, f: 'Function') -> bool:
        self.varID = None
        return self._update_imm(f)
    
    def desc(self, tokens=False):
        if tokens:
            imm = [('=', 'varID='), ('V', str(self.varID))]
        else:
            imm = f"   varID={self.varID}"
        
        return super().desc(tokens) + imm


class ConstInstr(InstrWithImm):
    """The `i32.const` instruction
    
    Instance variable:
    
      val: int              value of the constant
    
    """
    def __init__(self, mnemonic: str, opcode: bytes, pop: int, push: int):
        super().__init__(mnemonic, opcode, pop, push)
        self.val = None
    
    def _update_imm(self, f: 'Function') -> bool:
        if self.val is None:
            if f.bs is None:
                self.val = util.rnd_i32_0s1s()
            else:
                self.val = util.signed2unsigned(f.bs.next_LEB128(), 32)
        
        self.imm = util.LEB128(util.unsigned2signed(self.val, 32))
        return True
    
    def regen_imm(self, f: 'Function') -> bool:
        self.val = None
        return self._update_imm(f)
    
    def desc(self, tokens=False):
        if tokens:
            imm = [('=', 'val='), ('V', str(self.val))]
        else:
            imm = f"   val={self.val}"
        
        return super().desc(tokens) + imm


"""
The Reduced Instruction Set (RIS)
- - - - - - - - - - - - - - - - -

In terms of stack operands (pop - push):

  sources +1: local.get  i32.const
  
  neutral  0: block  loop  else  end  br  return  call  local.tee
              i32.load  i32.load8_u  i32.eqz  i32.clz  i32.ctz  i32.popcnt
  
    sinks -1: if  br_if  drop  local.set
              i32.eq  i32.ne  i32.lt_u  i32.gt_u  i32.le_u  i32.ge_u
              i32.add  i32.sub  i32.mul  i32.div_u  i32.rem_u
              i32.and  i32.or  i32.xor
              i32.shl  i32.shr_u  i32.rotl  i32.rotr
  
    sinks -2: select  i32.store  i32.store8
  
  sum of sources =  +2
  sum of sinks   = -28  [= (-1)*(4+6+5+3+4) + (-2)*(3)]


Emitting all instructions with equal frequencies leads to two problems:

  1. Control instruction etc get swamped by random arithmetic ones
  
  2. There are far too many operand sinks and nowhere near enough sources

In order to deal with Problem 2, let's increase the frequency of control 
and memory instructions (except local.get) say 4x for block starts and
2x for everything else.

This then increases the weighted sum of sinks to:

    sinks -1: 4x if  4x br_if  2x drop  2x local.set
              i32.eq  i32.ne  i32.lt_u  i32.gt_u  i32.le_u  i32.ge_u
              i32.add  i32.sub  i32.mul  i32.div_u  i32.rem_u
              i32.and  i32.or  i32.xor
              i32.shl  i32.shr_u  i32.rotl  i32.rotr
  
    sinks -2: 2x select  2x i32.store  2x i32.store8

  sum of sinks = (-1)*((4+4+2+2)+6+5+3+4) + (-2)*((2+2+2)) = -42

so let's pick 16x for the sources so the final balance is +32 vs -42.
"""
# the following instructions sometimes need to be used directly
# so need named variables outside of RIS_xxx
else_instr = ElseInstr('else', b'\x05', 0, 0)
end_instr = EndInstr('end', b'\x0b', 0, 0)
drop_instr = Instr('drop', b'\x1a', 1, 0)

# ( weight0, (instr00, instr01, ...),
#   weight1, (instr10, instr11, ...), ...)
RIS = (
    # operand sources
    16, (
        ConstInstr('i32.const i32', b'\x41', 0, 1),
        VarInstr('local.get x', b'\x20', 0, 1)
    ),
    
    # block starts
    4, (
        BlockInstr('block bt', b'\x02', 0, 0),
        BlockInstr('loop bt', b'\x03', 0, 0),
        BlockInstr('if bt', b'\x04', 1, 0)
    ),
    
    # end instruction (NB neutral for stack operands)
    # should be x12 to balance out block starts with as many block ends,
    # but need many more ends as fewer of them get accepted
    32, (end_instr,),
    
    # remainder of control instructions, and memory instructions
    2, (
        else_instr,
        BranchInstr('br l', b'\x0c', 0, 0),
        BranchInstr('br_if l', b'\x0d', 1, 0),
        ReturnInstr('return', b'\x0f', 0, 0),
        CallInstr('call x', b'\x10'),
        drop_instr,
        Instr('select', b'\x1b', 3, 1),
        VarInstr('local.set x', b'\x21', 1, 0),
        VarInstr('local.tee x', b'\x22', 1, 1),
        MemInstr('i32.load m', b'\x28', 1, 1),
        MemInstr('i32.load8_u m', b'\x2d', 1, 1),
        MemInstr('i32.store m', b'\x36', 2, 0),
        MemInstr('i32.store8 m', b'\x3a', 2, 0)
    ),
    
    # the great unwashed (aka artihmetic instructions)
    1, (
        Instr('i32.eqz', b'\x45', 1, 1),
        Instr('i32.clz', b'\x67', 1, 1),
        Instr('i32.ctz', b'\x68', 1, 1),
        Instr('i32.popcnt', b'\x69', 1, 1),
        Instr('i32.eq', b'\x46', 2, 1),
        Instr('i32.ne', b'\x47', 2, 1),
        Instr('i32.lt_u', b'\x49', 2, 1),
        Instr('i32.gt_u', b'\x4b', 2, 1),
        Instr('i32.le_u', b'\x4d', 2, 1),
        Instr('i32.ge_u', b'\x4f', 2, 1),
        Instr('i32.add', b'\x6a', 2, 1),
        Instr('i32.sub', b'\x6b', 2, 1),
        Instr('i32.mul', b'\x6c', 2, 1),
        Instr('i32.div_u', b'\x6e', 2, 1),
        Instr('i32.rem_u', b'\x70', 2, 1),
        Instr('i32.and', b'\x71', 2, 1),
        Instr('i32.or', b'\x72', 2, 1),
        Instr('i32.xor', b'\x73', 2, 1),
        Instr('i32.shl', b'\x74', 2, 1),
        Instr('i32.shr_u', b'\x76', 2, 1),
        Instr('i32.rotl', b'\x77', 2, 1),
        Instr('i32.rotr', b'\x78', 2, 1)
    )
)

RIS_instrs = [i for ins in RIS[1::2] for i in ins]
RIS_weights = [w for w, ins in zip(RIS[0::2], RIS[1::2]) for i in ins]
RIS_opcode2instr = {int.from_bytes(i.opcode, 'big'):i for i in RIS_instrs}
RIS_mnemonic2instr = {i.mnemonic:i for i in RIS_instrs}

class CodeBlock(Instr):
    """A block of WebAssembly code
    
    Used directly for `block` and `loop` blocks and as a base class for `if` 
    and `else` blocks (which are more complicated because `else` blocks are 
    treated as sub-blocks of `if` blocks)
    
    Behaves like a pseudo-instruction: opcode is None, but the rest of the 
    variables (i.e. mnemonic, pop, push, stack_after, fn_i and blk_i) 
    are defined and used, as are b() and append_to()
    
    Instance variables:
    
      targ: int             target number of operands on the stack
                              (`end` only accepted when stack size == targ)
      br_targ: int          target for `br`s branching down to this block
                              (=targ except for `loop`, where br_targ=0)
      temp_targ: int        temporary stack target for insertions etc
      content: list         instructions (and other blocks) within this block
      any_OK_after: int     index of the first `br` or `return` within content,
                              None => not issued yet; means "(almost) any 
                              instruction after this will be accepted"
      blocks: list          list of blocks within which this block is nested
                              (needed for `br`; includes self at the end)
      is_L0: bool           is this the Level 0 (i.e. function-level) block?
                              (more readable than doing len(blocks)==1)
      restr_end: bool       is the final `end` in this block restricted?
                              (if True, check Function.restr_end_OK first)
    """
    def __init__(self,
                 instr0: 'Instr',
                 parent: 'CodeBlock',
                 fn_targ: int = None):
        """Two ways of calling this constructor:
        
        1. Level 0 block (function level):
           
             CodeBlock(instr0=DummyFnStart(), parent=None, fn_targ=fn_targ)
        
        2. The standard blocks (`block` and `loop`):
           
             CodeBlock(instr0=BlockInstr(...), parent)
           
           [Also called by IfBlock.__init__(...)]
        
        Anything else will result in an error
        """
        if parent is None:
            # no parent, we're at Level 0 (function level)
            assert type(instr0) is DummyFnStart
            assert fn_targ is not None
            
            m = 'L0 block'
            super().__init__(mnemonic=m, opcode=None, pop=0, push=fn_targ)
            self.blk_i = 'L0'
            self.stack_after = self.push
            self.targ = fn_targ
            self.br_targ = fn_targ
            self.blocks = [self]
            self.is_L0 = True
            self.restr_end = True
        else:
            # have parent, we're at Level > 0
            assert type(instr0) is BlockInstr
            assert isinstance(parent, CodeBlock)
            assert fn_targ is None
            
            assert instr0.return_type in ('empty', 'i32')
            targ = 1 if instr0.return_type == 'i32' else 0
            
            m = f"'{instr0.mnemonic}' block"
            super().__init__(mnemonic=m, opcode=None,
                             pop=instr0.pop, push=targ)
            
            self.targ = targ
            self.br_targ = 0 if instr0.mnemonic == 'loop bt' else targ
            self.blocks = parent.blocks + [self]
            self.is_L0 = False
            self.restr_end = False
        
        self.temp_targ = None
        self.content = [instr0]
        self.any_OK_after = None
        pass
    
    def b(self):
        return b''.join([instr.b() for instr in self.content])
    
    def _copy(self):
        return self
    
    def set_fn_i(self, i: int) -> bool:
        self.fn_i = i
    
    def del_level(self, i: int) -> bool:
        del(self.blocks[i])
    
    def desc(self, tokens=False):
        if tokens:
            dummy = copy.copy(self.content[0])
            dummy.pop = self.pop
            dummy.push = self.push
            dummy.stack_after = self.stack_after
            return dummy.desc(tokens)
        else:
            return super().desc(tokens)
    
    def dump(self, tokens=False):
        spacer = ' '*3
        if tokens:
            start = 1
            NL = [('N', '')]
            if self.is_L0:
                spacer = []
            else:
                spacer = [(' ', spacer)]
        else:
            start = 0
            NL = ''
            if self.is_L0:
                spacer = ''
        
        if not self.is_L0:
            yield self.desc(tokens) + NL
        
        for i in self.content[start:]:
            if isinstance(i, CodeBlock):
                for s in i.dump(tokens):
                    yield spacer + s
            else:
                yield spacer + i.desc(tokens) + NL


class IfBlock(CodeBlock):
    """The `if` block
    
    Instance variables:
    
      need_else: bool         does this block *need* an `else`?
                                (`if` blocks with targ>0 must have an `else`)
      else_blk: 'ElseBlock'   `else` sub-block, None => not created yet
    
    """
    def __init__(self, instr0: 'Instr', parent: 'CodeBlock'):
        assert type(instr0) is BlockInstr
        assert instr0.mnemonic == 'if bt'
        
        super().__init__(instr0, parent)
        
        self.need_else = (self.targ > 0)
        self.else_blk = None
    
    def b(self):
        e = b'' if self.else_blk is None else self.else_blk.b()
        return super().b() + e
    
    def del_level(self, i: int) -> bool:
        super().del_level(i)
        if self.else_blk is not None:
            self.else_blk.del_level(i)
    
    def dump(self, tokens=False):
        yield from super().dump(tokens)
        if self.else_blk is not None:
            yield from self.else_blk.dump(tokens)
        


class ElseBlock(CodeBlock):
    """The `else` block
    
    Unlike all other blocks, the `else` block is not inserted into any 
    CodeBlock.content and instead is only contained in the parent `if`
    block's else_blk member
    
    Instance variables:
    
      if_blk: 'CodeBlock'   the parent `if` block
    
    """
    def __init__(self, instr0: 'Instr', parent: 'CodeBlock'):
        assert type(instr0) is ElseInstr
        assert type(parent) is IfBlock
        
        super(CodeBlock, self).__init__(mnemonic="'else' block", opcode=None,
                                        pop=parent.pop, push=parent.push)
        
        for attr in ('stack_after  blk_i  fn_i  '
                     'targ  br_targ  temp_targ  '
                     'is_L0 restr_end').split():
            setattr(self, attr, getattr(parent, attr))
        
        self.blocks = parent.blocks[:-1] + [self]
        self.if_blk = parent
        self.content = [instr0]
        self.any_OK_after = None
    
    def set_fn_i(self, i: int) -> bool:
        self.fn_i = i
        self.if_blk.fn_i = i


class Function:
    """A WebAssembly function
    
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
      wind_down: bool       if set, do not start any new blocks and only add
                              half the instructions that change the stack size
                              away from blk.targ (or blk.temp_targ)
      restr_end_OK: bool    OK to issue the final `end` for restricted blocks?
                              ('restricted' means CodeBlock.restr_end=True)
      creative_OK: bool     OK to creatively add/substitute instructions?
                              (e.g. `end` -> `else` in an `if` block that
                               needs an `else`, or add an `end` after a `br`)
      length: int           length in number of instructions
      bs: util.ByteStream   stream of bytes to parse (only used for parsing)
      index: list           flat list of all instructions
                              (see build_index() for details)
    
    """
    def __init__(self, gen0=None, parse=None):
        """Two ways of calling this constructor:
        
        1. Generate a new function from scratch ("generation 0"):
           
             Function(gen0=(fn_targ, length))
           
           where:
           
             targ: int          target for the Level 0 block
                                  (see CodeBlock for details)
             length: int        minimum length of the new function
                                  (length = number of instructions)
           
        2. Parse an existing function:
        
             Function(parse=(old_bytes, fn_targ))
           
           where:
           
             old_bytes: bytes   old code to mutate
             targ: int          target for the Level 0 block
                                  (see CodeBlock for details)
        
        Anything else will result in an error!
        """
        if gen0 is not None:
            assert parse is None
            fn_targ, length = gen0
            assert type(fn_targ) is int
            assert type(length) is int
        elif parse is not None:
            old_bytes, fn_targ = parse
            assert type(old_bytes) is bytes
            assert type(fn_targ) is int
        else:
            raise RuntimeError("need either 'gen0' or 'parse'")
        
        self.L0_blk = CodeBlock(instr0=DummyFnStart(), parent=None,
                                fn_targ=fn_targ)
        self.last_instr = self.L0_blk.content[-1]
        self.cur_blk = self.L0_blk
        self.wind_down = None
        self.restr_end_OK = None
        self.creative_OK = None
        self.length = 0
        self.bs = None
        self.index = None
        
        if gen0 is not None:
            self.generate(length)
        elif parse is not None:
            self.parse(old_bytes)
    
    def b(self) -> bytes:
        if self.cur_blk is None:
            return self.L0_blk.b()
        else:
            return None
    
    def dump(self, tokens=False):
        if tokens:
            return self.L0_blk.dump(tokens)
        else:
            print(f"length = {self.length}")
            print(*self.L0_blk.dump(tokens), sep="\n")
    
    def _append_instr(self, instr: 'Instr'):
        self.cur_blk.content.append(instr)
        self.last_instr = instr
    
    def _new_blk(self, instr0: 'Instr'):
        if instr0.mnemonic == 'if bt':
            new_blk = IfBlock(instr0=instr0, parent=self.cur_blk)
        elif instr0.mnemonic == 'else':
            new_blk = ElseBlock(instr0=instr0, parent=self.cur_blk)
        else:
            new_blk = CodeBlock(instr0=instr0, parent=self.cur_blk)
        
        if instr0.mnemonic == 'else':
            self.cur_blk.else_blk = new_blk
        else:
            new_blk.append_to(self)
        
        self.cur_blk = new_blk
        self.last_instr = instr0
    
    def _end_blk(self, instr: 'Instr'):
        self.cur_blk.content.append(instr)
        self.cur_blk.set_fn_i(instr.fn_i)
        self.last_instr = self.cur_blk
        
        if self.cur_blk.is_L0:
            self.cur_blk = None
        else:
            self.cur_blk = self.cur_blk.blocks[-2]
    
    def _generate_instr(self):
        instr = random.choices(RIS_instrs, weights=RIS_weights)[0]
        if instr.append_to(self):
            self.length += 1
    
    def generate(self, length: int):
        self.wind_down = False
        self.restr_end_OK = False
        self.creative_OK = True
        while self.length < length:
            self._generate_instr()
        
        self.wind_down = True
        self.restr_end_OK = True
        while self.cur_blk is not None:
            self._generate_instr()
    
    def parse(self, code: bytes):
        self.wind_down = False
        self.restr_end_OK = True
        self.creative_OK = False
        self.bs = util.ByteStream(code)
        while not self.bs.done():
            opcode = self.bs.next_i8()
            instr = RIS_opcode2instr[opcode]
            if instr.append_to(self):
                self.length += 1
            else:
                raise RuntimeError(f"syntax error at instr {self.length}")
        
        self.cur_blk = None
        self.bs = None
    
    def verify(self) -> bool:
        """Typically called after a mutation
        
        Verifies that the function passes validation, in which case returns 
        True (and False otherwise)
        """
        try:
            new = Function(parse=(self.b(), self.L0_blk.targ))
        except RuntimeError:
            return False
        
        return True
    
    def build_index(self):
        """Build self.index = a flat list of all instructions
        
        Each entry is:
        
            (instr, blk, parent_blk)
        
        where the following is always true:
        
            instr = blk.content[instr.blk_i]
        
        and then depending on whether whether blk is an ElseBlock:
            
            blk = parent_blk.content[blk.blk_i].else_blk   # ElseBlocks
            blk = parent_blk.content[blk.blk_i]            # all other cases
        
        and further when blk.is_L0 is True, parent_blk is None.
        
        The index starts with the initial `DummyFnStart` at index[0]
        and ends with the terminal `end` (at the very end of the function)
        at index[self.length]
        
        The following is True for all index entries:
        
            index[j][0].fn_i == j-1
        
        Remember to set self.index to None when no longer needed (eg just 
        before actually mutating the function) so there are no hanging 
        references impeding the garbage collector
        """
        def add_block(blk, parent_blk=None):
            for instr in blk.content:
                if isinstance(instr, CodeBlock):
                    add_block(instr, blk)
                else:
                    self.index.append((instr, blk, parent_blk))
            
            if type(blk) is IfBlock and blk.else_blk is not None:
                add_block(blk.else_blk, parent_blk)
        
        self.index = []
        add_block(self.L0_blk)
    
    def check_index(self):
        """sanity check for self.index"""
        for instr, blk, parent_blk in self.index:
            assert instr == blk.content[instr.blk_i]
            if blk.is_L0:
                assert parent_blk is None
            else:
                if type(blk) is ElseBlock:
                    assert blk == blk.if_blk.else_blk
                    assert blk == parent_blk.content[blk.blk_i].else_blk
                else:
                    assert blk == parent_blk.content[blk.blk_i]
        
        for j in range(len(self.index)):
            assert self.index[j][0].fn_i == j-1
    
    def random_region(self, length: int):
        """Find a random region of at least `length` instructions
        consisting of non-block instructions and whole blocks
        
        The region does *not* need to be stack-neutral (pop-push != 0) 
        
        Returns (blk, start, end) where the region consists of instructions
        blk.content[start] through to blk.content[end-1]
        
        Returns (None, None, None) if there is no such region
        
        The initial dummy `DummyFnStart` and the terminal `end` instruction
        at the very end of the function are never part of the region
        """
        for L in range(length, self.length):
            # [self.length] is the terminal `end` at the end of the function
            # we want the last start to be L before that
            # +1 because range(3) is (0, 1, 2)
            starts = list(range(1, self.length-L+1))
            random.shuffle(starts)
            
            for start in starts:
                s, s_blk, s_pblk = self.index[start]        # s = start
                e, e_blk, e_pblk = self.index[start+L-1]    # e = end
                
                # too complicated for how rare it is, skipped
                if type(s) is ElseInstr:
                    continue
                
                if type(s) is BlockInstr: s, s_blk = s_blk, s_pblk
                if type(e) is EndInstr: e, e_blk = e_blk, e_pblk
                
                if s_blk != e_blk:
                    continue
                
                return (s_blk, s.blk_i, e.blk_i+1)
        
        return (None, None, None)
    
    def gtor_random_stack_neutral_region(self,
                                         length_min: int,
                                         length_max: int = None,
                                         i_start: int = None,
                                         i_end: int = None):
        """Find a random stack-neutral (pop-push = 0) region of at least 
        `length_min` instructions consisting of non-block instructions and 
        whole blocks
        
        Optionally can also specify `length_max` as the maximum number of 
        instructions, and `i_start` and `i_end` as the subregion to which 
        the search is restricted
        
        Yields (blk, start, end) where the stack-neutral region found 
        consists of instructions blk.content[start] through to 
        blk.content[end-1] and where i_start <= start and end <= i_end
        
        With default settings (i_start = 1, i_end = self.length), the initial 
        dummy `DummyFnStart` at self.index[0] (which is not counted as part of 
        self.length) and the terminal `end` instruction at the very end of the 
        function at self.index[self.length] will never be part of the region
        """
        if length_max is None: length_max = self.length - 1
        if i_start is None: i_start = 1
        if i_end is None: i_end = self.length
        assert i_start <= i_end
        
        for L in range(length_min, length_max+1):
            # [i_end-1] is the last instruction to include in the region
            # we want the last start to be L-1 before that
            # +1 because range(3) is (0, 1, 2)
            starts = list(range(i_start, i_end-L+1))
            random.shuffle(starts)
            for start in starts:
                p, p_blk, p_pblk = self.index[start-1]      # p = previous
                s, s_blk, s_pblk = self.index[start]        # s = start
                e, e_blk, e_pblk = self.index[start+L-1]    # e = end
                
                # too complicated for how rare it is, skipped
                if type(s) is ElseInstr:
                    continue
                
                if type(p) is EndInstr: p, p_blk = p_blk, p_pblk
                if type(s) is BlockInstr: s, s_blk = s_blk, s_pblk
                if type(e) is EndInstr: e, e_blk = e_blk, e_pblk
                
                assert p_blk == s_blk
                if s_blk != e_blk or p.stack_after != e.stack_after:
                    continue
                
                yield (s_blk, s.blk_i, e.blk_i+1)
    
    def random_stack_neutral_region(self, length):
        gtor = self.gtor_random_stack_neutral_region(length)
        try:
            out = next(gtor)
        except StopIteration:
            return (None, None, None)
        
        return out
    
    def random_stack_neutral_Uregion(self, length):
        """Find a random stack-neutral (pop-push = 0) U-region of at least 
        `length` instructions consisting of non-block instructions and whole 
        blocks, where a 'U-region' is defined as two segments of code within 
        the same block separated by at least one instruction and/or sub-block 
        that is not part of the U-region. The two segments of the U-region 
        taken together are stack-neutral.
        
        Returns (blk, start1, end1, start2, end2) where the stack-neutral 
        U-region consists of instructions blk.content[start1:end1] and 
        blk.content[start2:end2] (NB [0:3] = [0, 1, 2])
                
        Returns (None, None, None, None, None) when there is no such region
        
        The initial dummy `DummyFnStart` at self.index[0] (which is not 
        counted as part of self.length) and the terminal `end` instruction at 
        the very end of the function at self.index[self.length] will never be 
        part of the U-region
        """
        # one instruction, skipped region, one instruction = two instructions
        if length < 2: length = 2 
        
        #               .-- exclude the `end` at the very end of the function
        #               |   .-- at least one instruction in skipped region
        #               |   |   .-- range(3) = (0, 1, 2)
        # self.length - 1 - 1 + 1
        for L_main in range(length, self.length-1):
            
            #               .-- exclude the terminal `end` at end of function
            #               |      .-- already covered by L_main
            #               |      |     .-- range(3) = (0, 1, 2)
            # self.length - 1 + L_main + 1
            L_skips = list(range(1, self.length - L_main))
            random.shuffle(L_skips)
            for L_skip in L_skips:
                L = L_main + L_skip
                gtor_main = self.gtor_random_stack_neutral_region(L, L)
                for blk, start, end in gtor_main:
                    # [start] is the first instruction in the main region
                    # [end-1] is the last instruction in the main region
                    # but fn_i for a block is the fn_i of its last instruction
                    # so [start] --> [start-1]+1 to get the first one
                    si = blk.content[start-1].fn_i + 1
                    ei = blk.content[end-1].fn_i
                    
                    tot_length = ei - si + 1
                    assert tot_length == L_main + L_skip
                    
                    # want [si+1] ... [ei-1]
                    # +1 to map fn_i --> self.index[i]
                    # +1 for i_end because range(3) = (0, 1, 2)
                    i_start = si + 2
                    i_end = ei + 1
                    assert i_start <= i_end
                    
                    p = (L_skip, L_skip, i_start, i_end)
                    gtor_skip = self.gtor_random_stack_neutral_region(*p)
                    for skip_blk, skip_start, skip_end in gtor_skip:
                        if skip_blk == blk:
                            return (blk, start, skip_start, skip_end, end)
        
        return (None, None, None, None, None)
    
    def random_reachable_instr(self):
        """Find a random reachable instruction
        
        Reachable instruction = one before, or at, its block's any_OK_after
        (if there is one; if not, all instruction in the block are reachable)
        
        Returns (instr, blk) where
        
            instr = blk.content[instr.blk_i]
        
        If the instruction is an `end`, returns the whole block (along with
        its parental block). May return the DummyFnStart dummy instruction 
        at the start of the function, but never the `end` at the very end 
        of the function.
        """ 
        reachable = [i for i in self.index
                     if (i[1].any_OK_after is None or
                         i[0].blk_i <= i[1].any_OK_after)]
        
        del(reachable[-1])  # get rid of the final `end`
        instr, blk, parent_blk = random.choice(reachable)
        if type(instr) is EndInstr:
            instr, blk = blk, parent_blk
        
        return (instr, blk)
    
    def random_instr_filter(self, filter_fn):
        """Find a randomly chosen instruction from a filtered list
        of instructions that match filter_fn(instr)
        
        Returns the full index entry for that instruction, i.e.
        
            (instr, blk, parent_blk, is_else_blk)
        
        """
        return random.choice([i for i in self.index if filter_fn(i[0])])
    
    instr_classes = (
        {'i32.load m', 'i32.load8_u m'},
        {'i32.store m', 'i32.store8 m'},
        {'i32.eqz', 'i32.clz', 'i32.ctz', 'i32.popcnt'},
        {'i32.eq', 'i32.ne', 'i32.lt_u', 'i32.gt_u', 'i32.le_u', 'i32.ge_u'},
        {'i32.add', 'i32.sub', 'i32.mul', 'i32.div_u', 'i32.rem_u'},
        {'i32.and', 'i32.or', 'i32.xor'},
        {'i32.shl', 'i32.shr_u', 'i32.rotl', 'i32.rotr'})
    
    def mutator_mut_instr(self, length: int) -> bool:
        """Mutate a single non-control instruction to an instruction 
        in the same class and with the same net effect on the stack
        (i.e. same pop-push). Immediates (if any) are retained.
        
        The `length` parameter is ignored.
        """
        def filter_fn(i):
            return (type(i) is MemInstr or
                    (type(i) is Instr and i.mnemonic.startswith('i32.')))
        
        try:
            i = self.random_instr_filter(filter_fn)[0]
        except IndexError:
            return False
        
        # there's always exactly 1 match, any exception here is a bug
        cl = next(c for c in Function.instr_classes if i.mnemonic in c)
        
        new_i = RIS_mnemonic2instr[random.choice(list(cl - {i.mnemonic}))]
        
        # UGLY HAAAAACK ... kids, don't do this at home!
        i.mnemonic = new_i.mnemonic
        i.opcode = new_i.opcode
        return True
    
    def mutator_mut_imm(self, length: int) -> bool:
        """Mutate a single instruction immediate by regenerating it from 
        scratch. Only used with instructions where changing the immediate
        will have no effect on the stack.
        
        The `length` parameter is ignored.
        """
        def filter_fn(i):
            return hasattr(i, 'regen_imm')
        
        try:
            i = self.random_instr_filter(filter_fn)[0]
        except IndexError:
            return False
        
        old_imm = i.imm
        for _ in range(10):
            if not i.regen_imm(self):
                continue
            
            if i.imm != old_imm:
                return True
        
        return False
    
    def mutator_del_blk(self, length: int) -> bool:
        """Delete a single block instruction (`block`, `loop` or `if`) and its 
        corresponding `end`. Instructions inside the block are retained and 
        simply moved down a level. For `if` blocks (which have pop=1),
        a single `drop` instruction is prepended. For an `if` block with
        an `else` section, one of the two sections is picked at random and 
        retained, and the other deleted.  Branch instructions inside the block 
        are adjusted and those branching to the deleted block are removed 
        altogether, with (1) `br_if` replaced by a `drop`, (2) for `br` the 
        instruction itself and the rest of the block is deleted and then 
        either several `drop` instructions are appended (if stack size > block 
        target) or several newly generated ones (if stack size < block target) 
        in order to get to get to the target level. 
        
        The `length` parameter is ignored.
        """
        def filter_blk(i):
            return type(i) is BlockInstr
        
        try:
            instr, blk, parent_blk = self.random_instr_filter(filter_blk)
        except IndexError:
            return False
        
        extra = []
        if type(blk) is IfBlock:
            self.cur_blk = parent_blk
            self.last_instr = parent_blk.content[blk.blk_i - 1]
            drop = copy.copy(drop_instr)
            assert drop._finish_init(self)
            extra = [drop]
            
            if blk.else_blk is not None and random.choice((True, False)):
                blk = blk.else_blk
                instr = blk.content[0]
        
        level = len(blk.blocks) - 1
        skip_end = 1 if type(blk.content[-1]) is EndInstr else 0
        instrs = self.index[blk.content[0].fn_i + 2 :
                            blk.content[-1].fn_i + 2 - skip_end]
        
        blk.del_level(level)
        for bl_instr, bl_blk, _ in instrs:
            if type(bl_instr) is not BlockInstr: continue
            bl_blk.del_level(level)
        
        for br_instr, br_blk, _ in instrs:
            if type(br_instr) is not BranchInstr:
                continue
            
            # dest_level is the *old* absolute level, 0 means 'Level 0'
            # (should have been len(blocks)-1 but called del_level(...) above)
            dest_level = len(br_blk.blocks) - br_instr.dest
            
            if dest_level > level:
                continue
            
            if br_instr not in br_blk.content:
                # this instruction clearly followed an earlier `br`
                # so it's already been deleted, skipping
                continue
            
            self.cur_blk = br_blk
            self.last_instr = br_blk.content[br_instr.blk_i - 1]
            
            if dest_level < level:
                br_instr.dest -= 1
                assert br_instr._update_imm(self)
            elif dest_level == level:
                if br_instr.mnemonic == 'br_if l':
                    drop = copy.copy(drop_instr)
                    assert drop._finish_init(self)
                    br_blk.content[br_instr.blk_i] = drop
                elif br_instr.mnemonic == 'br l':
                    if br_instr.blk_i <= br_blk.any_OK_after:
                        br_blk.any_OK_after = None
                    
                    end = br_blk.content[-1]
                    del(br_blk.content[br_instr.blk_i:])
                    
                    if self.last_instr.stack_after < br_blk.targ:
                        # HACK: length=0 so no new blocks
                        # (new blocks can't reach br_blk after their `end`
                        #  because it's been deleted, so get an infinite loop)
                        assert self.mutator_ins(length=0, blk=br_blk,
                                                start=len(br_blk.content),
                                                targ=br_blk.targ)
                    else:
                        while self.last_instr.stack_after > br_blk.targ:
                            drop_instr.append_to(self)
                    
                    # `if` blocks with an `else` section haven't got an `end`
                    if type(end) is EndInstr:
                        br_blk.content.append(end)
        
        dst = blk.blk_i
        del(parent_blk.content[dst])
        parent_blk.content[dst:dst] = (extra +
            blk.content[1 : len(blk.content)-skip_end])
        
        self.cur_blk = None
        return True
    
    def mutator_ins_blk(self, length: int) -> bool:
        """Insert a single new block instruction (`block`, `loop` or `if`) and 
        its corresponding `end` around several existing non-block instructions 
        and/or whole blocks. Branch instructions inside the newly created 
        block are adjusted. For `if` blocks (which have pop=1), a series of 
        random instructions is prepended to increase the stack by 1. For `if` 
        blocks that require an `else` section, the existing instructions are 
        randomly allocated to either the `if` or the `else` section and the 
        other section is filled with random new instructions.
        
        The new block will envelop at least `length` existing instructions.
        """
        return False
    
    def mutator_del_else(self, length: int) -> bool:
        """Find an `if` block that has an `else` section but does not need it, 
        and delete the `else` section.
        
        The `length` parameter is ignored.
        """
        return False
    
    def mutator_ins_else(self, length: int) -> bool:
        """Find an `if` block that does not have an `else` section and 
        generate an `else` section for it filled with random new instructions.
        
        The new `else` section will be at least `length` instructions long.
        """
        return False
    
    def mutator_swap_else(self, length: int) -> bool:
        """Find an `if` block that has an `else` section and swap the content 
        of the two sections.
        
        The `length` parameter is ignored.
        """
        return False
    
    def mutator_del(self, length: int = None,
                    blk: CodeBlock = None,
                    start: int = None,
                    end: int = None) -> bool:
        """Delete several non-block instructions and/or whole blocks, but
        do not add any new ones. The deleted region must be stack-neutral
        (pop-push = 0).
        
        At least `length` instructions are deleted.
        """
        if blk is None or start is None or end is None:
            if length is None:
                raise RuntimeError("Need either length or (blk, start, end)!")
            
            blk, start, end = self.random_stack_neutral_region(length)
        
        if blk is None:
            return False
        
        # this boldly ignores a few errors due to CodeBlock.any_OK_after
        # (too complicated to handle correctly for how rare they are)
        del(blk.content[start:end])
        
        # heads up: self.length is now wrong (but no longer needed)
        return True
    
    def mutator_delU(self, length) -> bool:
        """Delete several non-block instructions and/or whole blocks, but do 
        not add any new ones. The deleted code forms two segments within the 
        same block separated by at least one instruction and/or block that is 
        retained. Taken together the two deleted segments are stack-neutral 
        (pop-push = 0).
        
        At least `length` instructions are deleted.
        """
        blk, s1, e1, s2, e2 = self.random_stack_neutral_Uregion(length)
        
        if blk is None:
            return False
        
        # this boldly ignores a few errors due to CodeBlock.any_OK_after
        # (too complicated to handle correctly for how rare they are)
        # as well as stack size dropping below zero in [e2:s1]
        del(blk.content[s2:e2])
        del(blk.content[s1:e1])
        
        # heads up: self.length is now wrong (but no longer needed)
        return True
    
    def mutator_ins(self, length: int,
                    blk: CodeBlock = None,
                    start: int = None,
                    targ: int = None) -> bool:
        """Insert several new non-block instructions and/or whole blocks, 
        leaving existing instructions intact. The inserted region must be 
        stack-neutral (pop-push = 0).
        
        At least `length` instructions are inserted.
        """
        if blk is None or start is None or targ is None:
            prev_instr, blk = self.random_reachable_instr()
            start = prev_instr.blk_i+1
            targ = prev_instr.stack_after
        else:
            prev_instr = blk.content[start-1]
        
        old_length = self.length
        old_content_len = len(blk.content)
        old_blk = copy.copy(blk)
        rest = blk.content[start:]
        del(blk.content[start:])
        
        blk.restr_end = True
        if blk.any_OK_after is not None and start <= blk.any_OK_after:
            blk.any_OK_after = None
        
        self.cur_blk = blk
        self.last_instr = prev_instr
        self.wind_down = False
        self.restr_end_OK = False
        self.creative_OK = True
        while self.length - old_length < length:
            self._generate_instr()
        
        self.wind_down = True
        blk.temp_targ = targ
        while self.cur_blk != blk or self.last_instr.stack_after != targ:
            self._generate_instr()
        
        blk.content += rest
        blk.temp_targ = None
        blk.restr_end = old_blk.restr_end
        if blk.any_OK_after is None and old_blk.any_OK_after is not None:
            d = len(blk.content) - old_content_len
            blk.any_OK_after = old_blk.any_OK_after + d
        
        self.cur_blk = None
        return True
    
    def mutator_dup(self, length: int) -> bool:
        """Duplicate a series of non-block instructions and/or whole blocks, 
        i.e. create a copy immediately after the original. The duplicated 
        region must be stack-neutral (pop-push = 0).
        
        At least `length` instructions are duplicated.
        """
        blk, start, end = self.random_stack_neutral_region(length)
        if blk is None:
            return False
        
        blk.content[end:end] = copy.deepcopy(blk.content[start:end])
        return True
    
    def mutator_cp(self, length: int) -> bool:
        """Copy a series of non-block instructions and/or whole blocks 
        to a random new location elsewhere in the function. The copied region
        must be stack neutral (pop-push = 0).
        
        At least `length` instructions are copied.
        """
        src_blk, start, end = self.random_stack_neutral_region(length)
        if src_blk is None:
            return False
        
        prev_instr, dst_blk = self.random_reachable_instr()
        i = prev_instr.blk_i+1
        
        # ignoring errors due to the stack dipping below zero in some cases
        dst_blk.content[i:i] = copy.deepcopy(src_blk.content[start:end])
        return True
    
    def mutator_mv(self, length: int) -> bool:
        """Move a series of non-block instructions and/or whole blocks
        to a random new location elsewhere in the function. The moved region
        must be stack neutral (pop-push = 0).
        
        At least `length` instructions are moved.
        """
        return False
    
    def mutator_reorder(self, length: int) -> bool:
        """Move a series of non-block instructions and/or whole blocks
        to a random new location within their current block. The moved region
        does not need to be stack-neutral (pop-push != 0), but stack size
        may never dip below zero.
        
        At least `length` instructions are moved.
        """
        return False
    
    def mutator_swap(self, length: int) -> bool:
        """Find two series of non-block instructions and/or whole blocks that 
        have the same net effect on the stack (pop-push) and swap them.
        Stack size may never dip below zero.
        
        The longer of two regions (or both if they are of equal length) is 
        at least `length` instructions long.
        """
        return False
    
    def mutator_overwrite(self, length: int) -> bool:
        """Overwrite several non-block instructions and/or whole blocks with 
        newly generated instructions. The overwritten region does *not* need
        to be stack-neutral (pop-push != 0).
        
        At least `length` instructions get overwritten. The newly generated 
        region may be shorter (or longer) than the one it replaced.
        """
        blk, start, end = self.random_region(length)
        if blk is None:
            return False
        
        targ = blk.content[end-1].stack_after
        return (self.mutator_del(blk=blk, start=start, end=end) and
                self.mutator_ins(1, blk=blk, start=start, targ=targ))
    
    def mutate(self, method: str, length: int) -> bool:
        """Mutate the current function
        
        Params:
        
             method: str        Function.mutator_{method}() to apply
             length: int        minimum length of any changes
                                  (length = number of instructions;
                                   this option is ignored by some methods)
        """
        assert type(method) is str
        assert type(length) is int
        
        self.build_index()
        mutator_fn = getattr(self, 'mutator_' + method)
        retval = mutator_fn(length)
        self.index = None
        
        return retval
