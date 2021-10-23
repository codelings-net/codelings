import util
import random
from copy import copy


class Instr:
    """
    A single instruction within a WebAssembly (or CodeLang) program
    
    Used directly for simple instructions without an immediate and as a base 
    class for more complicated instructions
    
    Instruction objects are initialised in two steps:
    
    1. A list of all available instructions is created as the class variable
       Function.RIS_instrs, but this leaves stack_after and i set to None
    
    2. When append_to() is called on these Instr objects, it creates a copy of 
       itself, completes the initialisation *of the copy* and then appends
       it to the current CodeBlock
    
    Instance variables:
    
      mnemonic: str         instruction mnemonic ('i32.const', 'if' etc) 
      opcode: bytes         binary opcode
      pop: int              number of params popped from the operand stack
      push: int             number of results pushed onto the operand stack
      stack_after: int      number of operands on the stack after this instr
      i: int                this is instruction number i in the byte string
    
    """
    def __init__(self, mnemonic: str, opcode: bytes, pop: int, push: int):
        self.mnemonic = mnemonic
        self.opcode = opcode
        self.pop = pop
        self.push = push
        self.stack_after = None
        self.i = None
    
    def b(self) -> bytes:
        return self.opcode
    
    def _append_OK(self, f: 'Function') -> bool:
        return self.pop <= f.last_instr.stack_after or f.cur_blk.any_OK
    
    def _copy_action(self, f: 'Function'):
        return copy(self)
    
    def _update_stack(self, f: 'Function'):
        self.stack_after = f.last_instr.stack_after - self.pop + self.push
    
    def _append_action(self, f: 'Function'):
        f._append_instr(self)
    
    def append_to(self, f: 'Function') -> bool:
        """returns True if an instruction was appended, False otherwise"""
        
        if not self._append_OK(f):
            return False
        
        cp = self._copy_action(f)
        if cp is None:
            return False
        
        cp.i = f.last_instr.i + 1
        cp._update_stack(f)
        cp._append_action(f)
        return True
    
    def desc(self):
        return f"{self.i:3}: {self.mnemonic:20} -{self.pop} +{self.push} " \
            f"[{self.stack_after:3}]"


class InstrWithImm(Instr):
    """
    Base class for instructions with immediates (bytes that follow the opcode)
    
    NB imm is initialised in Step 2 (see the docstring for Instr on how
    instruction objects are initialised)
    
    Instance variables:
    
      imm: bytes            binary immediate(s)
    
    """
    def __init__(self, mnemonic: str, opcode: bytes, pop: int, push: int):
        super().__init__(mnemonic, opcode, pop, push)
        self.imm = None
    
    def b(self) -> bytes:
        return self.opcode + self.imm
    
    def _get_imm(self, f: 'Function') -> bool:
        """
        returns True if imm will pass validation, False otherwise
        
        *** TO BE OVERRIDDEN IN DERIVED CLASSES ***
        
        """
        raise NotImplementedError
    
    def _copy_action(self, f: 'Function'):
        cp = copy(self)
        if not cp._get_imm(f):
            del cp
            return None
        else:
            return cp


class FnStart(Instr):
    """
    Dummy intruction inserted at the start of each function
    
    Instr.append_to() assumes that there is a last_instr. All blocks begin with 
    the block instruction (`block` | `loop` | `if`), but that leaves the 
    function-level block, which is where we insert this dummy.
    """ 
    def __init__(self):
        super().__init__(mnemonic='FnStart', opcode=b'', pop=0, push=0)
        self.stack_after = 0
        self.i = -1


class BlockInstr(InstrWithImm):
    """
    Block instructions: `block`, `loop` and `if`
    
    Instance variable:
    
      return_type: str      'empty' | 'i32'
    
    """
    def __init__(self, mnemonic: str, opcode: bytes, pop: int, push: int):
        super().__init__(mnemonic, opcode, pop, push)
        assert mnemonic in ('block bt', 'loop bt', 'if bt')
        self.return_type = None
    
    def _append_OK(self, f: 'Function') -> bool:
        return (self.pop <= f.last_instr.stack_after or f.cur_blk.any_OK) and \
            f.new_blks_OK
    
    def _update_stack(self, f: 'Function'):
        self.stack_after = 0
    
    def _get_imm(self, f: 'Function') -> bool:
        options = (b'\x40', b'\x7f')
        if f.bs is None:
            self.imm = random.choice(options)
        else:
            self.imm = f.bs.next_b()
            assert self.imm in options
            
        self.return_type = 'empty' if self.imm == b'\x40' else 'i32'
        return True
    
    def _append_action(self, f: 'Function'):
        f._new_blk(instr0=self)
    
    def desc(self):
        return super().desc() + f"   return_type={self.return_type}"


class ElseInstr(Instr):
    """The `else` instruction"""
    
    def _append_OK(self, f: 'Function') -> bool:
        return f.cur_blk.if_block and f.cur_blk.else_index is None and \
            (f.last_instr.stack_after == f.cur_blk.targ or f.cur_blk.any_OK)
    
    def _update_stack(self, f: 'Function'):
        self.stack_after = 0
    
    def _append_action(self, f: 'Function'):
        f._append_instr(self)
        f.cur_blk.need_else = False
        f.cur_blk.else_index = len(f.cur_blk.content) - 1
        f.cur_blk.any_OK = False


class EndInstr(Instr):
    """The `end` instruction"""
    
    def _append_OK(self, f: 'Function') -> bool:
        # `if` blocks with targ>0 need an `else`, otherwise we'd get an error
        if f.cur_blk.need_else:
            if f.creative_OK:
                return Function.else_instr._append_OK(f)
            else:
                return False
        
        return (not f.cur_blk.is_L0 or f.L0_end_OK) and \
            (f.last_instr.stack_after == f.cur_blk.targ or f.cur_blk.any_OK)
    
    def _append_action(self, f: 'Function'):
        f._end_blk(self)
    
    def append_to(self, f: 'Function') -> bool:
        """returns True if an instruction was appended, False otherwise"""
        
        if f.cur_blk.need_else and f.creative_OK:
            return Function.else_instr.append_to(f)
        
        return super().append_to(f)


class BranchInstr(InstrWithImm):
    """
    The `br` and `br_if` instructions
    
    Instance variable:
    
      dest: int             destination block to which the code will branch
                              (0 = current block, 1 = the block one down, etc)
    """
    def __init__(self, mnemonic: str, opcode: bytes, pop: int, push: int):
        super().__init__(mnemonic, opcode, pop, push)
        assert mnemonic in ('br l', 'br_if l')
        self.dest = None
    
    def _append_OK(self, f: 'Function') -> bool:
        # the bulk of testing is done in _get_imm() once the dest is known
        return self.mnemonic != 'br l' or not f.cur_blk.is_L0 or f.L0_end_OK
    
    def _update_imm(self):
        self.imm = util.uLEB128(self.dest)
    
    def _get_imm(self, f: 'Function') -> bool:
        if f.bs is None:
            dest = random.randrange(len(f.cur_blk.blocks))
        else:
            dest = f.bs.next_uLEB128()
            assert dest < len(f.cur_blk.blocks)
        
        br_targ = f.cur_blk.blocks[-1 - dest].br_targ
        if f.last_instr.stack_after - self.pop < br_targ:
            if not f.cur_blk.any_OK: return False
        
        self.dest = dest
        self._update_imm()
        return True
    
    def _append_action(self, f: 'Function'):
        f._append_instr(self)
        if self.mnemonic == 'br l':
            f.cur_blk.any_OK = True
            if f.creative_OK:
                Function.end_instr.append_to(f)   # add an `end` instruction
    
    def desc(self):
        return super().desc() + f"   dest={self.dest}"


class ReturnInstr(Instr):
    """The `return` instruction"""
    
    def _append_OK(self, f: 'Function') -> bool:
        targ = f.cur_blk.blocks[0].targ
        return (not f.cur_blk.is_L0 or f.L0_end_OK) and \
            (targ <= f.last_instr.stack_after or f.cur_blk.any_OK)
    
    def _append_action(self, f: 'Function'):
        f._append_instr(self)
        f.cur_blk.any_OK = True
        if f.creative_OK:
            Function.end_instr.append_to(f)       # add an `end` instruction


class CallInstr(InstrWithImm):
    """The `call` instruction"""
    
    def __init__(self, mnemonic: str, opcode: bytes):
        """
        pop and push depend on what function is called, so only initialised
        in Step 2 (see the docstring for Instr for details)
        """
        super().__init__(mnemonic, opcode, None, None)
    
    def _append_OK(self, f: 'Function') -> bool:
        """
        stack size can only be tested once the signature of the function
        to be called is known, i.e. once imm is known
        """
        return True
    
    def _get_imm(self, f: 'Function') -> bool:
        # XXX TODO
        # we are assuming only one function with zero params and zero results
        # and are hard-coding a call to this function (=function no. 0)
        # i.e. a recursive call to itself
        fnID = b'\x00'
        if f.bs is None:
            self.imm = fnID
        else:
            self.imm = f.bs.next_b()
            assert self.imm == fnID
        
        self.pop = 0
        self.push = 0
        
        # zero params = always passes validation
        return True


class MemInstr(InstrWithImm):
    """
    Memory instructions:
    
      `i32.load`, `i32.load8_u`, `i32.store`, `i32.store8`
    
    Instance variables:
    
      align: int            the align part of memarg (see the WebAssembly spec)
      offset: int           the offset part of memarg (see the WebAssembly spec)
    
    where the alignment (align) is a promise to the VM that:
    
      (baseAddr + memarg.offset) mod 2^memarg.align == 0
    
    "Unaligned access violating that property is still allowed and must succeed 
    regardless of the annotation. However, it may be substantially slower on 
    some hardware."
    """
    def __init__(self, mnemonic: str, opcode: bytes, pop: int, push: int):
        super().__init__(mnemonic, opcode, pop, push)
        self.align = None
        self.offset = None
    
    def _update_imm(self):
        self.imm = util.uLEB128(self.align) + util.uLEB128(self.offset)
    
    def _get_imm(self, f: 'Function') -> bool:
        if f.bs is None:
            # hard-coded poor align (slow but always works)
            # mostly small random offset (mostly 0x00 and 0x01)
            self.align = 0x00
            self.offset = util.rnd_i32_0s1s()
        else:
            self.align = f.bs.next_uLEB128()
            self.offset = f.bs.next_uLEB128()
        
        self._update_imm()
        return True
    
    def desc(self):
        return super().desc() + f"   align={self.align} offset={self.offset}"


class VarInstr(InstrWithImm):
    """
    Instructions dealing with local (and eventually global as well) variables: 
    
      `local.get`, `local.set` and `local.tee`
    
    Instance variable:
    
      varID: int            ID of variable to be operated on
    
    """
    def __init__(self, mnemonic: str, opcode: bytes, pop: int, push: int):
        super().__init__(mnemonic, opcode, pop, push)
        self.varID = None
    
    def _update_imm(self):
        self.imm = util.uLEB128(self.varID)
    
    def _get_imm(self, f: 'Function') -> bool:
        # XXX TODO: hard-coded 4 locals
        n_vars = 4
        if f.bs is None:
            self.varID = random.randrange(n_vars)
        else:
            self.varID = f.bs.next_uLEB128()
            assert self.varID <= n_vars
        
        self._update_imm()
        return True
    
    def desc(self):
        return super().desc() + f"   varID={self.varID}"


class ConstInstr(InstrWithImm):
    """
    The `i32.const` instruction
    
    Instance variable:
    
      val: int              value of the constant
    
    """
    def __init__(self, mnemonic: str, opcode: bytes, pop: int, push: int):
        super().__init__(mnemonic, opcode, pop, push)
        self.val = None
    
    def _update_imm(self):
        # XXX TODO: weirdly this only seems to work up to 31 bits?!?
        self.imm = util.uLEB128(self.val)
    
    def _get_imm(self, f: 'Function') -> bool:
        if f.bs is None:
            self.val = util.rnd_i32_0s1s()
        else:
            self.val = f.bs.next_uLEB128()
        
        self._update_imm()
        return True
    
    def desc(self):
        return super().desc() + f"   val={self.val}"


class CodeBlock(Instr):
    """
    A block of WebAssembly code
    
    Behaves like a pseudo-instruction: opcode is None, but the rest of the 
    variables (i.e. pop, push, stack_after) are defined and used, as are
    b() and append_to()
    
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
      is_L0: bool           is this the Level 0 (function level) block?
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
            
            m = 'L0 block'
            super().__init__(mnemonic=m, opcode=None, pop=0, push=targ)
            self.i = -1
            self.stack_after = targ
            
            self.targ = targ
            self.br_targ = targ
            self.if_block = False
            self.need_else = False
            self.blocks = [self]
            self.is_L0 = True
        else:
            # have parent, we're at Level > 0
            assert type(instr0) is BlockInstr
            assert type(parent) is CodeBlock
            assert targ is None
            
            assert instr0.return_type in ('empty', 'i32')
            targ = 1 if instr0.return_type == 'i32' else 0
            
            m = f"'{instr0.mnemonic}' block"
            super().__init__(mnemonic=m, opcode=None, pop=instr0.pop, push=targ)
            
            self.targ = targ
            self.br_targ = 0 if instr0.mnemonic == 'loop bt' else targ
            self.if_block = (instr0.mnemonic == 'if bt')
            self.need_else = (self.if_block and self.targ > 0)
            self.blocks = parent.blocks + [self]
            self.is_L0 = False
        
        self.else_index = None
        self.content = [instr0]
        self.any_OK = False
     
    def b(self):
        return b''.join([instr.b() for instr in self.content])
    
    def _copy_action(self, f: 'Function'):
        return self
    
    def dump(self):
        if self.is_L0:
            spacer = ''
        else:
            yield super().desc()
            spacer = ' '*4
        
        for i in self.content:
            try:
                for s in i.dump():
                    yield spacer + s
            except:
                yield spacer + i.desc()


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
      length: int           length in number of instructions
      bs: util.ByteStream   stream of bytes to parse (only used for parsing)
    
    """
    def __init__(self, gen0=None, mutate=None):
        """
        Two ways of calling this constructor:
        
        1. Generate a new function from scratch ("generation 0"):
        
             Function(gen0=(targ, length))
        
           where:
           
             targ: int          target for the Level 0 block
                                  (see CodeBlock for details)
             length: int        minimum length of the new function
                                  (length = number of instructions)
           
        2. Parse and mutate an existing function:
        
             Function(mutate=(old_bytes, targ, method, length))
           
           where:
           
             old_bytes: bytes   old code to mutate
             targ: int          target for the Level 0 block
                                  (see CodeBlock for details)
             method: str        Function.mutator_{method}() to apply
             length: int        minimum length of any changes
                                  (length = number of instructions;
                                   this option is ignored by some methods)
        
        Anything else will result in an error!
        """
        if gen0 is not None:
            assert mutate is None
            targ, length = gen0
            assert type(targ) is int
            assert type(length) is int
        elif mutate is not None:
            old_bytes, targ, method, length = mutate
            assert type(old_bytes) is bytes
            assert type(targ) is int
            assert type(method) is str
            assert type(length) is int
            
            mutator_fn = getattr(self, 'mutator_' + method)
        else:
            raise RuntimeError("need either 'gen0' or 'mutate'")
        
        self.L0_blk = CodeBlock(instr0=FnStart(), parent=None, targ=targ)
        self.last_instr = self.L0_blk.content[-1]
        self.cur_blk = self.L0_blk
        self.new_blks_OK = None
        self.L0_end_OK = None
        self.creative_OK = None
        self.length = 0
        self.bs = None
        
        if gen0 is not None:
            self.generate(length)
        elif mutate is not None:
            self.parse(old_bytes)
            
            ### XXX TODO: What are we going to do when this (vvv) returns False,
            ### i.e. could not find a mutation that would pass validation?
            mutator_fn(length)
    
    """
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
    # `else` and `end` need to be called directly when creative_OK is set, 
    # so need named variables outside of RIS_source
    else_instr = ElseInstr('else', b'\x05', 0, 0)
    end_instr = EndInstr('end', b'\x0b', 0, 0)
    
    # ( weight0, (instr00, instr01, ...),
    #   weight1, (instr10, instr11, ...), ...)
    RIS = (
        # operand sources
        16, (
            ConstInstr('i32.const', b'\x41', 0, 1),
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
            Instr('drop', b'\x1a', 1, 0),
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
    
    def _append_instr(self, instr: 'Instr'):
        self.cur_blk.content.append(instr)
        self.last_instr = instr
    
    def _new_blk(self, instr0: 'Instr'):
        new_blk = CodeBlock(instr0=instr0, parent=self.cur_blk)
        new_blk.append_to(self)
        self.cur_blk = new_blk
        self.last_instr = instr0
    
    def _end_blk(self, instr: 'Instr'):
        self.cur_blk.content.append(instr)
        self.cur_blk.i = instr.i
        self.last_instr = self.cur_blk
        
        if self.cur_blk.is_L0:
            self.cur_blk = None
        else:
            self.cur_blk = self.cur_blk.blocks[-2]
    
    def _generate_instr(self):
        instr = random.choices(self.RIS_instrs, weights=self.RIS_weights)[0]
        if instr.append_to(self):
            self.length += 1
    
    def generate(self, length: int):
        self.new_blks_OK = True
        self.L0_end_OK = False
        self.creative_OK = True
        while self.length < length and self.cur_blk is not None:
            self._generate_instr()
        
        self.new_blks_OK = False
        self.L0_end_OK = True
        while self.cur_blk is not None:
            self._generate_instr()
    
    def b(self) -> bytes:
        if self.cur_blk is None: 
            return self.L0_blk.b()
        else:
            return None
    
    def build_index(self):
        pass
    
    def parse(self, code: bytes):
        print('parse')
        
        self.new_blks_OK = True
        self.L0_end_OK = True
        self.creative_OK = False
        self.bs = util.ByteStream(code)
        
        while not self.bs.done():
            opcode = self.bs.next_int()
            instr = self.RIS_opcode2instr[opcode]
            #print(type(instr))
            instr.append_to(self)
        
        print('sanity check, please remove when done testing')
        assert self.b() == code
        
        print(*self.L0_blk.dump(), sep="\n")
        
        self.cur_blk = None
    
    def mutator_tweak_imm(self, length: int) -> bool:
        """
        Mutate a single instruction immediate by changing it a little.
        
        The `length` parameter is ignored.
        """
        return False
    
    def mutator_regen_imm(self, length: int) -> bool:
        """
        Mutate a single instruction immediate by regenerating it from scratch.
        
        The `length` parameter is ignored.
        """
        return False
    
    def mutator_tweak_instr(self, length: int) -> bool:
        """
        Mutate a single non-control instruction to an instruction in the same 
        class and with the same net effect on the stack (i.e. same pop-push). 
        Immediates (if any) are retained.
        
        The classes are as follows:
        
          (i32.load m,  i32.load8_u m)
          (i32.store m,  i32.store8 m)
          (i32.eqz,  i32.clz,  i32.ctz,  i32.popcnt)
          (i32.eq,  i32.ne,  i32.lt_u,  i32.gt_u,  i32.le_u,  i32.ge_u)
          (i32.add,  i32.sub,  i32.mul,  i32.div_u,  i32.rem_u)
          (i32.and,  i32.or,  i32.xor)
          (i32.shl,  i32.shr_u,  i32.rotl,  i32.rotr).
        
        The `length` parameter is ignored.
        """
        return False
    
    def mutator_regen_instr(self, length: int) -> bool:
        """
        Mutate a single non-control instruction to a different non-control 
        instruction with the same net effect on the stack (i.e. same pop-push). 
        Immediates (if any) are regenerated from scratch.
        
        The control instructions (which are skipped) are as follows:
        
          `block`  `loop`  `else`  `end`  `br`  `return`  `call`.
        
        All other instructions are in play.
        
        The `length` parameter is ignored.
        """
        return False
    
    def mutator_del_block(self, length: int) -> bool:
        """ Delete a single block instruction (`block`, `loop` or `if`) + its 
        corresponding `end`. Instructions inside the block are retained and 
        simply moved down a level. (For an `if` block with an `else` section, 
        one of the two sections is picked at random and retained and the other 
        deleted.) Branch instructions inside the block are adjusted and those 
        branching to the deleted block are removed altogether (`br_if` replaced 
        by a `drop`). For `if` blocks (which have pop=1) a single `drop` 
        instruction is prepended.
        
        The `length` parameter is ignored.
        """
        return False
    
    def mutator_ins_block(self, length: int) -> bool:
        """
        Insert a single new block instruction (`block`, `loop` or `if`) + its 
        corresponding `end` around several existing non-block instructions 
        and/or whole blocks. Branch instructions inside the newly created block 
        are adjusted. For `if` blocks (which have pop=1), a series of random 
        instructions is prepended to increase the stack by 1. For `if` blocks 
        that require an `else` section, the existing instructions are randomly 
        allocated to either the `if` or the `else` section and the other 
        section is filled with random new instructions.
        
        The new block will envelop at least `length` existing instructions.
        """
        return False
    
    def mutator_del_else(self, length: int) -> bool:
        """
        Find an `if` block that has an `else` section but does not need it, and 
        delete the `else` section.
        
        The `length` parameter is ignored.
        """
        return False
    
    def mutator_ins_else(self, length: int) -> bool:
        """
        Find an `if` block that does not have an `else` section and generate an 
        `else` section for it filled with random new instructions.
        
        The new `else` section will be at least `length` instructions long.
        """
        return False
    
    def mutator_del(self, length: int) -> bool:
        """
        Delete several non-block instructions and/or whole blocks, but do not 
        add any new ones. The deleted region must be stack neutral 
        (pop-push = 0).
        
        At least `length` instructions are deleted.
        """
        print('mutator_del')
        return False
    
    def mutator_ins(self, length: int) -> bool:
        """
        Insert several new non-block instructions and/or whole blocks, leaving 
        existing instructions intact. The inserted region must be stack neutral 
        (pop-push = 0).
        
        At least `length` instructions are inserted.
        """
        return False
    
    def mutator_dup(self, length: int) -> bool:
        """
        Duplicate a series of non-block instructions and/or whole blocks, i.e. 
        create a copy immediately after the original. The duplicated region 
        must be stack neutral (pop-push = 0).
        
        At least `length` instructions are duplicated.
        """
    
    def mutator_cp(self, length: int) -> bool:
        """
        Copy a series of non-block instructions and/or whole blocks to a random 
        new location elsewhere in the function. The copied region must be stack 
        neutral (pop-push = 0).
        
        At least `length` instructions are copied.
        """
        return False
    
    def mutator_mv(self, length: int) -> bool:
        """
        Move a series of non-block instructions and/or whole blocks to a random 
        new location elsewhere in the function. The moved region must be stack 
        neutral (pop-push = 0).
        
        At least `length` instructions are moved.
        """
        return False
    
    def mutator_reorder(self, length: int) -> bool:
        """
        Move a series of non-block instructions and/or whole blocks to a random 
        new location within their current block. The moved region does *not* 
        need to be stack neutral (pop-push != 0).
        
        At least `length` instructions are moved.
        """
        return False
    
    def mutator_overwrite(self, length: int) -> bool:
        """
        Overwrite several non-block instructions and/or whole blocks with newly 
        generated instructions. The overwritten region does *not* need to be 
        stack neutral (pop-push != 0).
        
        At least `length` instructions get overwritten. The newly generated 
        region may be shorter than the one it replaced.
        """
        return False
