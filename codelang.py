import util
import random
from copy import copy


class Instr:
    """
    A single instruction within a WebAssembly program
    
    Used directly for simple instructions without an immediate and as a base 
    class for more complicated instructions
    
    Instruction objects are initialised in two steps:
    
    1. A list of all available instructions is created as the class variable
       Function.RIS_instrs, but this leaves stack_after set to None
    
    2. When append_to() is called on these Instr objects, it creates a copy of 
       itself, completes the initialisation *of the copy* and then appends
       it to the current CodeBlock
    
    Instance variables:
    
      opcode: bytes         binary opcode
      pop: int              number of params popped from the operand stack
      push: int             number of results pushed onto the operand stack
      stack_after: int      number of operands on the stack after this instr
    
    """
    def __init__(self, opcode: bytes, pop: int, push: int):
        self.opcode = opcode
        self.pop = pop
        self.push = push
        self.stack_after = None
    
    def to_bytes(self):
        return self.opcode
    
    def _append_OK(self, f: 'Function') -> bool:
        return self.pop <= f.last_instr.stack_after or f.cur_blk.any_OK
    
    def _update_stack(self, f: 'Function'):
        self.stack_after = f.last_instr.stack_after - self.pop + self.push
    
    def _append(self, f: 'Function'):
        f._append_instr(self)
    
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
    def __init__(self, opcode: bytes, pop: int, push: int):
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
    def __init__(self):
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
        return (self.pop <= f.last_instr.stack_after or f.cur_blk.any_OK) and \
            f.new_blks_OK
    
    def _update_stack(self, f: 'Function'):
        self.stack_after = 0
    
    def _get_imm(self, f: 'Function') -> bool:
        self.imm = random.choice((b'\x40', b'\x7f'))
        self.return_type = 'empty' if self.imm == b'\x40' else 'i32'
        return True
    
    def _append(self, f: 'Function'):
        f._new_blk(instr0=self)


class ElseInstr(Instr):
    """The `else` instruction"""
    
    def _append_OK(self, f: 'Function') -> bool:
        return f.cur_blk.if_block and f.cur_blk.else_index is None and \
            (f.last_instr.stack_after == f.cur_blk.targ or f.cur_blk.any_OK)
    
    def _update_stack(self, f: 'Function'):
        self.stack_after = 0
    
    def _append(self, f: 'Function'):
        f._append_instr(self)
        f.cur_blk.need_else = False
        f.cur_blk.else_index = len(f.cur_blk.content) - 1


class EndInstr(Instr):
    """The `end` instruction"""
    
    def _append_OK(self, f: 'Function') -> bool:
        # `if` blocks with targ>0 need an `else`, otherwise we'd get an error
        if f.cur_blk.need_else:
            if f.creative_OK:
                return Function.else_instr._append_OK(f)
            else:
                return False
        
        return (not f.cur_blk.L0 or f.L0_end_OK) and \
            (f.last_instr.stack_after == f.cur_blk.targ or f.cur_blk.any_OK)
    
    def _append(self, f: 'Function'):
        f._end_blk(self)
    
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
    def __init__(self, opcode: bytes, pop: int, push: int, branch_type: str):
        super().__init__(opcode, pop, push)
        assert branch_type in ('br', 'br_if')
        self.branch_type = branch_type
        self.dest = None
    
    def _append_OK(self, f: 'Function') -> bool:
        # the bulk of the testing is done in _get_imm() once the imm is known
        return self.branch_type != 'br' or not f.cur_blk.L0 or f.L0_end_OK
    
    def _get_imm(self, f: 'Function') -> bool:
        dest = random.randrange(len(f.cur_blk.blocks))
        br_targ = f.cur_blk.blocks[-1 - dest].br_targ
        if f.last_instr.stack_after - self.pop < br_targ:
            if not f.cur_blk.any_OK:
                return False
        
        self.dest = dest
        self.imm = util.uLEB128(dest)
        return True
    
    def _append(self, f: 'Function'):
        f._append_instr(self)
        if self.branch_type == 'br':
            f.cur_blk.any_OK = True
            if f.creative_OK:
                Function.end_instr.append_to(f)   # add an `end` instruction


class ReturnInstr(Instr):
    """The `return` instruction"""
    
    def _append_OK(self, f: 'Function') -> bool:
        targ = f.cur_blk.blocks[0].targ
        return (not f.cur_blk.L0 or f.L0_end_OK) and \
            (targ <= f.last_instr.stack_after or f.cur_blk.any_OK)
    
    def _append(self, f: 'Function'):
        f._append_instr(self)
        f.cur_blk.any_OK = True
        if f.creative_OK:
            Function.end_instr.append_to(f)       # add an `end` instruction


class CallInstr(InstrWithImm):
    """The `call` instruction"""
    
    def __init__(self, opcode: bytes):
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
        self.imm = b'\x00' + util.uLEB128(util.rnd_i32_0s1s())
        return True


class VarInstr(InstrWithImm):
    """
    Instructions dealing with local (and eventually global as well) variables: 
        `local.get`, `local.set` and `local.tee`
    """
    
    def _get_imm(self, f: 'Function') -> bool:
        # XXX TODO: hard-coded 4 locals
        self.imm = util.uLEB128(random.randrange(0x04))
        return True


class ConstInstr(InstrWithImm):
    """The `i32.const` instruction"""
    
    def _get_imm(self, f: 'Function') -> bool:
        # XXX TODO: weirdly this only seems to work up to 31 bits?!?
        self.imm = util.uLEB128(util.rnd_i32_0s1s())
        return True


class CodeBlock(Instr):
    """
    A block of WebAssembly code
    
    Behaves like a pseudo-instruction: opcode is None, but the rest of the 
    variables (i.e. pop, push, stack_after) are defined and used, as are
    to_bytes() and append_to()
    
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
      n_instr: int          number of instructions
    """
    def __init__(self, targ: int):
        self.L0_blk = CodeBlock(instr0=FnStart(), parent=None, targ=targ)
        self.last_instr = self.L0_blk.content[-1]
        self.cur_blk = self.L0_blk
        self.new_blks_OK = None
        self.L0_end_OK = None
        self.creative_OK = None
        self.n_instr = 0
    
    # `else` and `end` need to be called directly when creative_OK is set, 
    # so need named variables outside of RIS_source
    else_instr = ElseInstr(b'\x05', 0, 0)
    end_instr = EndInstr(b'\x0b', 0, 0)
    
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
    # ( weight0, (instr00, instr01, ...),
    #   weight1, (instr10, instr11, ...), ...)
    RIS = (
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
        # but need many more ends as fewer of them get accepted
        32, (end_instr,),
        
        # remainder of control instructions, and memory instructions
        2, (
            else_instr,                             # else
            BranchInstr(b'\x0c', 0, 0, 'br'),       # br l
            BranchInstr(b'\x0d', 1, 0, 'br_if'),    # br_if l
            ReturnInstr(b'\x0f', 0, 0),             # return
            CallInstr(b'\x10'),                     # call x
            Instr(b'\x1a', 1, 0),                   # drop
            Instr(b'\x1b', 3, 1),                   # select
            VarInstr(b'\x21', 1, 0),                # local.set x
            VarInstr(b'\x22', 1, 1),                # local.tee x
            MemInstr(b'\x28', 1, 1),                # i32.load m
            MemInstr(b'\x2d', 1, 1),                # i32.load8_u m
            MemInstr(b'\x36', 2, 0),                # i32.store m
            MemInstr(b'\x3a', 2, 0),                # i32.store8 m
        ),
        
        # the great unwashed (aka artihmetic instructions)
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
    
    RIS_instrs = [i for ins in RIS[1::2] for i in ins]
    RIS_weights = [w for w, ins in zip(RIS[0::2], RIS[1::2]) for i in ins]
    RIS_opcode2instr = {i.opcode: i for i in RIS_instrs}
    
    def _append_instr(self, instr: 'Instr'):
        self.cur_blk.content.append(instr)
        self.last_instr = instr
    
    def _new_blk(self, instr0: 'Instr'):
        new_blk = CodeBlock(instr0=instr0, parent=self.cur_blk)
        new_blk._update_stack(self)
        self.cur_blk = new_blk
        self.last_instr = instr0
    
    def _end_blk(self, instr: 'Instr'):
        self.cur_blk.content.append(instr)
        self.last_instr = self.cur_blk
        
        if self.cur_blk.L0:
            self.cur_blk = None
        else:
            self.cur_blk = self.cur_blk.blocks[-2]
    
    def _generate_instr(self):
        instr = random.choices(self.RIS_instrs, weights=self.RIS_weights)[0]
        if instr.append_to(self):
            self.n_instr += 1

    def generate(self, n_instr: int):
        self.new_blks_OK = True
        self.L0_end_OK = False
        self.creative_OK = True
        while self.n_instr < n_instr and self.cur_blk is not None:
            self._generate_instr()
        
        self.new_blks_OK = False
        self.L0_end_OK = True
        while self.cur_blk is not None:
            self._generate_instr()
   
    def to_bytes(self) -> bytes:
        if self.cur_blk is None: 
            return self.L0_blk.to_bytes()
        else:
            return None
    
    def parse(self, code: bytes) -> bool:
        pass
