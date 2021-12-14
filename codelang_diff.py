import colorama

import difflib


class IndelPrinter:
    html_header = ('<html>\n<body>\n<pre style="display: inline; '
                   'white-space: pre-wrap; word-wrap: break-word; '
                   'font-size: 110%; line-height: 1.3em; '
                   'color: #000000;">')
    html_footer = '</pre>\n</body>\n</html>'
    html_end_span = '</span>'
    html_new_span = ('<span style="background-color: %s; '
                     'padding-bottom: 0.2em;">')
    
    def __init__(self, outformat: str, width: int = 80):
        if outformat == 'ANSI':
            self.colour = {'M': colorama.Style.RESET_ALL,
                           'd': colorama.Back.RED,
                           'i': colorama.Back.GREEN}
            self.last_mode = 'M'
        elif outformat == 'HTML':
            self.colour = {'M': self.html_new_span % ('#FFFFFF'),
                           'd': self.html_new_span % ('#FF8888'),
                           'i': self.html_new_span % ('#88FF88')}
            self.last_mode = None
            print(self.html_header)
        else:
            raise RuntimeError(f"Unknown outpur format '{outformat}'")
        
        self.outformat = outformat
        self.width = width
        self.line_so_far = 0
    
    def indel_print(self, mode, s: str):
        if mode not in ('M', 'd', 'i'):
            raise RuntimeError(f"Unknown mode '{mode}'")
        
        if self.last_mode != mode:
            if self.outformat == 'HTML' and self.last_mode is not None:
                print(self.html_end_span, end='')
            
            print(self.colour[mode], end='')
        
        print(s, end='')
        self.last_mode = mode
        self.line_so_far += len(s)
    
    def newline(self):
        if self.outformat == 'ANSI' and self.last_mode != 'M':
            print(self.colour['M'])
            print(self.colour[self.last_mode], end='')
        else:
            print()
        
        self.line_so_far = 0
    
    def EOF(self):
        """end of file - called after each codeling"""
        if self.outformat == 'ANSI':
            self.indel_print('M', '')
        elif self.outformat == 'HTML':
            self.indel_print('M', ' ' * self.width)
        
        self.newline()
    
    def EOT(self):
        """end of transmission - called at the very end"""
        if self.outformat == 'HTML':
            print(self.html_footer)


class CodelangDiff:
    def __init__(self, prtr: 'IndelPrinter', new_f, old_f=None):
        """Arguments:
        
          prtr      where to send the diff
          new_f     new codelang function in token format
          old_f     old codelang function in token format
        
        where the token format produced by codelang.Instr.desc(token=True)
        looks like eg
        
          [('M', 'local.get x'), ('-', '0'), ('+', '1'), ('S', ' 1'),
           ('=', 'varID='), ('V', '1'), ('N', ''), ...]
        
        where the the token types are as follows:
        
          ' '       spacer
          'M'       mnemonic
          '-'       pop from stack
          '+'       push onto stack
          'S'       stack size after the instruction (Instr.stack_after)
          '='       immediate name
          'V'       immediate value
          'N'       new line
        
        so the list would get translated into:
        
          'local.get x              -0   +1   [ 1]      varID=1\n...'
        
        """
        self.posn = {'M': 0, '-': 25, '+': 30, 'S': 35, '=': 45, 
                     'N': prtr.width}
        self.spacer = False
        self.last_type = None   # type = {spacer, mnemonic, pop, ...}
        self.last_mode = None   # mode = {match, insertion, deletion}
        self.init_spaces = 0
        self.prtr = prtr
        
        self.diff(new_f, old_f)
    
    def printer(self, mode, content):
        for i_type, i in content:
            if i_type in self.posn:
                if i_type == 'N':
                    so_far = self.prtr.line_so_far
                else:
                    so_far = self.prtr.line_so_far - self.init_spaces
                
                new_spaces = self.posn[i_type] - so_far
                if not self.spacer or len(self.spacer) < new_spaces:
                    self.spacer = ' ' * new_spaces
            
            if self.spacer:
                if i_type == 'N':
                    self.prtr.indel_print(mode, self.spacer)
                elif self.last_type != i_type:
                    if ('M' in (mode, self.last_mode) or 
                            (mode == 'd' and self.last_mode == 'i')):
                        self.prtr.indel_print('M', self.spacer)
                    else:
                        self.prtr.indel_print(mode, self.spacer)
                
                self.spacer = False
            
            if i_type == ' ':               # spacer
                if self.prtr.line_so_far == self.init_spaces:
                    self.init_spaces += len(i)
                self.prtr.indel_print(mode, i)
            if i_type == 'M':               # mnemonic
                self.prtr.indel_print(mode, i)
            elif i_type in ('-', '+'):      # pop, push
                self.prtr.indel_print(mode, i_type + i)
            elif i_type == 'S':             # stack size (aka stack_after)
                self.prtr.indel_print(mode, '[' + i + ']')
            elif i_type in ('='):           # immediate name
                self.prtr.indel_print(mode, i)
            elif i_type == 'V':             # immediate value
                self.prtr.indel_print(mode, i)
                self.spacer = ' '
            elif i_type == 'N':             # new line
                self.prtr.newline()
                self.init_spaces = 0
                
            self.last_type = i_type
            self.last_mode = mode
    
    @staticmethod
    def expand_replaces(diff, old_f, new_f):
        for item in diff:
            tag, old_s, old_e, new_s, new_e = item  # _s = start, _e = end
            if tag != 'replace':
                yield item
                continue
            
            if ((old_e-old_s == 1 or new_e-new_s == 1) and
                old_f[old_s][0] == new_f[new_s][0] and
                old_f[old_s][0] in ('S', 'V')):
                # special case of a matching singleton, e.g.
                # 
                #   <replace> a1 b2 c3 <with> a2 </replace>
                # 
                # expanded to:
                # 
                #   <del> a1 </del> <ins> a2 </ins> <del> b2 c3 </del>
                # 
                yield ['delete', old_s, old_s+1, new_s, new_s]
                yield ['insert', old_s+1, old_s+1, new_s, new_s+1]
                
                if old_e - old_s > 1:
                    yield ['delete', old_s+1, old_e, new_s+1, new_s+1]
                
                if new_e - new_s > 1:
                    yield ['insert', old_e, old_e, new_s+1, new_e]
            else:
                # standard expansion, e.g.
                # 
                #   <replace> a1 b2 <with> c3 </replace>
                # 
                # expanded to:
                # 
                #   <del> a1 b2 </del> <ins> c3 </ins>
                # 
                yield ['delete', old_s, old_e, new_s, new_s]
                yield ['insert', old_e, old_e, new_s, new_e]
    
    @staticmethod
    def tidy_up_isolated_dels(diff: list, old_f):
        #          0      1          2        3          4
        # item = (tag, old_start, old_end, new_start, new_end)
        for i in range(1, len(diff)-1):
            if diff[i][0]=='delete' and diff[i-1][0]==diff[i+1][0]=='equal':
                # misassigned ambiguous starts, e.g. change:
                # 
                #   spacer  <del> mnemonic ... newline
                #   spacer </del>
                # 
                # to:
                # 
                #    <del> spacer mnemonic ... newline
                #   </del> spacer
                # 
                j = 0
                prev_old_start = diff[i-1][1]
                old_start, old_end = diff[i][1:3]
                while (prev_old_start <= old_start-j-1 and
                       old_f[old_start-j-1] == old_f[old_end-j-1] and
                       old_f[old_start-j-1][0] != 'N'):
                    j += 1
                
                if j > 0:
                    diff[i-1][2] -= j       # old_end
                    diff[i-1][4] -= j       # new_end
                    diff[i][1:] = [val-j for val in diff[i][1:]]
                    diff[i+1][1] -= j       # old_start
                    diff[i+1][3] -= j       # new_start
                
                # misassigned ambiguous ends, e.g. change:
                # 
                #   mnemonic1 ... stack1  <del> newline
                #   mnemonic2 ... stack2 </del> newline
                # 
                # to:
                # 
                #          mnemonic1 ... stack1 newline
                #    <del> mnemonic2 ... stack2 newline
                #   </del>
                # 
                j = 0
                next_old_end = diff[i+1][2]
                while (old_end+j < next_old_end and
                       old_f[old_start+j] == old_f[old_end+j]):
                    j += 1
                    if old_f[old_start+(j-1)][0] == 'N': break
                
                if j > 0:
                    diff[i-1][2] += j       # old_end
                    diff[i-1][4] += j       # new_end
                    diff[i][1:] = [val+j for val in diff[i][1:]]
                    diff[i+1][1] += j       # old_start
                    diff[i+1][3] += j       # new_start
        return diff
    
    @staticmethod
    def tidy_up_dels_missing_final_NL(diff: list, old_f):
        yield diff[0]
        
        #          0      1          2        3          4
        # item = (tag, old_start, old_end, new_start, new_end)
        for i in range(1, len(diff)-1):
            # if a deletion is missing a final newline, if possible steal it
            # from itself by breaking it into two
            # 
            # e.g. change:
            # 
            #   mnemonic1 ... stack1  <del> imms newline
            #   mnemonic2 ... stack2 </del> newline
            # 
            # to:
            # 
            #          mnemonic1 ... stack1 <del> imms </del> newline
            #    <del> mnemonic2 ... stack2 newline
            #   </del>
            # 
            tag, old_start, old_end, new, _ = diff[i]
            next_tag = diff[i+1][0]
            if (tag == 'delete' and next_tag == 'equal' and
                old_f[old_end][0] == 'N'):
                
                deletion = enumerate(old_f[old_end-1:old_start-1:-1])
                try:
                    d = next(i for i, item in deletion if item[0] == 'N')
                except StopIteration:
                    yield diff[i]
                    continue
                
                yield ['delete', old_start, old_end-d-1, new, new]
                yield ['equal', old_end-d-1, old_end-d, new, new+1]
                yield ['delete', old_end-d, old_end+1, new+1, new+1]
                
                diff[i+1][1] += 1       # old_start
                diff[i+1][3] += 1       # new_start
            else:
                yield diff[i]
        
        if len(diff) > 1:
            yield diff[-1]
    
    def diff(self, new_f, old_f=None):
        if old_f is None:
            printer('M', new_f)
            return
        
        d = difflib.SequenceMatcher(None, old_f, new_f).get_opcodes()
        d = [list(item) for item in d]
        d = CodelangDiff.expand_replaces(d, old_f, new_f)
        d = CodelangDiff.tidy_up_isolated_dels(list(d), old_f)
        d = CodelangDiff.tidy_up_dels_missing_final_NL(d, old_f)
        
        for tag, old_start, old_end, new_start, new_end in d:
            if tag == 'delete':
                self.printer('d', old_f[old_start:old_end])
            elif tag == 'equal':
                self.printer('M', old_f[old_start:old_end])
            elif tag == 'insert':
                self.printer('i', new_f[new_start:new_end])
            else:
                raise RuntimeError(f"uknown tag '{tag}'")
        
        self.prtr.EOF()
