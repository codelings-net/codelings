#!/usr/bin/env python

import codelang
import codelang_diff
import codeling
import config
import util

from types import SimpleNamespace as sns

import argparse
import collections
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


# set to True by SIGINT_handler() (called when the user presses Ctrl-C)
STOPPING = False


def SIGINT_handler(sig, frame):
    global STOPPING  # signal handler: globals need to be explicitly declared
    STOPPING = True


def stop_check(cdl_gtor) -> 'codeling.Codeling':
    for cdl in cdl_gtor:
        if STOPPING:
            print(' Caught SIGINT, stopping. Waiting for jobs to finish.',
                  file=sys.stderr)
            break
        else:
            yield cdl

def _init_pool_worker():
    # ignore SIGINT in pool workers
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _cdl_score(cdl: 'codeling.Codeling'):
    return cdl.score()


def score_Codelings(cfg: 'config.Config', cdl_gtor):
    if cfg.baseline_run:
        thresh = 'baseline run'
        baseline_results = {}
    else:
        if cfg.thresh is None:
            thresh = 'no threshold (all codelings rejected)'
        else:
            thresh = 'diff ' if cfg.diff else ''
            thresh += f"thresh={cfg.thresh:#x}"
    
    print(f"# Release {cfg.release}, Codeling.score_{cfg.scfn}(), {thresh}")
    
    if not (cfg.baseline_run or cfg.keep_all):
        print(f"# Reading '{cfg.indir}' to keep new codelings unique ... ",
              end='')
        seen_code = {}
        for json_fname in all_json_fnames(cfg.indir):
            cdl = codeling.Codeling(cfg, json_fname=json_fname)
            seen_code[cdl.json['code']] = json_fname
        
        print('done')
    
    t_start = time.time()
    print('# Scoring started:', util.nice_now())
        
    print('#')
    print("# All times below are in microseconds, the scores are in hex")
    
    IDlen = len(cfg.runid) + 13 if cfg.runid is not None else 20
    times = 't_gen t_valid t_run t_score'.split()
    hs = [f"{s:>7}" for s in (*times, *'fuel score status'.split(), 'n_acc ')]
    print(f"{'# ID':{IDlen}}", *hs, 'description', sep="\t")
    
    n_scored, n_accepted = 0, 0
    n_scored_prev, t_prev = 0, t_start
    
    with multiprocessing.Pool(cfg.nproc, _init_pool_worker) as p:
        # `.imap` because `.map` converts the iterable to a list
        # `_unordered` because don't care about order, really
        # *** when debugging use `map` instead for cleaner error messages ***
        #for r in map(_cdl_score, cdl_gtor):
        for r in p.imap_unordered(_cdl_score, cdl_gtor, chunksize=20):
            n_scored += 1
            
            if r.status == 'accept' and not (cfg.baseline_run or cfg.keep_all):
                if r.code in seen_code:
                    r.status = 'reject'
                    r.desc = f"duplicate of '{seen_code[r.code]}'"
                    os.remove(r.json_fname)
                    os.remove(util.json2wasm(r.json_fname))
                else:
                    seen_code[r.code] = r.json_fname
            
            if r.status == 'accept':
                n_accepted += 1
            
            micros = [round(1e6 * getattr(r, t)) for t in times]
            ts = [f"{s:>7}" for s in (*micros, r.fuel, f"{r.score:x}",
                                      r.status, f"{n_accepted} ")]
            print(f"{r.ID:{IDlen}}", *ts, r.desc, sep="\t")
            
            if cfg.baseline_run:
                baseline_results[r.json_fname] = r
            
            if n_scored % 100_000 == 0:
                if n_scored == 100_000:
                    print(f"{'time':23}",
                          *[f"{s:>15}" for s in ('n_scored/1e5',
                                                 'n_accepted',
                                                 'scored/hour')],
                          sep="\t", file=sys.stderr)
                
                t_now = time.time()
                thrpt = (n_scored - n_scored_prev) / (t_now - t_prev) * 3600
                n_scored_prev, t_prev = n_scored, t_now
                print(util.nice_now(),
                      *[f"{i:>15d}" for i in (round(n_scored / 1e5),
                                              n_accepted)],
                      f"{thrpt:>15.2e}",
                      sep="\t", file=sys.stderr)
    
    print('# Scoring finished:', util.nice_now())
    print(f"# Scoring throughput: {n_scored/(time.time()-t_start)*3600:.2e} "
          f"scored/hour")
    
    if cfg.baseline_run:
        return baseline_results


def all_json_fnames(d: str):
    return sorted(glob.glob(os.path.join(d, '*.json')))


def gtor_score(cfg: 'config.Config') -> 'codeling.Codeling':
    for json_fname in all_json_fnames(cfg.indir):
        yield codeling.Codeling(cfg, json_fname=json_fname,
                                wasm_fname=util.json2wasm(json_fname),
                                deferred=True)


def gtor_gen0(cfg: 'config.Config', Ls, N: int) -> 'codeling.Codeling':
    cdl_i = 0
    for i in range(N):
        for L in Ls:
            ID = f"{cfg.runid}-{cdl_i:012}"
            cdl_i += 1
            yield codeling.Codeling(cfg, gen0=(ID, L), deferred=True)


def gtor_mutate(cfg: 'config.Config', Ls, N: int, 
                baseline_results) -> 'codeling.Codeling':
    cdl_i = 0
    for i in range(N):
        for json_fname in all_json_fnames(cfg.indir):
            for L in Ls:
                ID = f"{cfg.runid}-{cdl_i:012}"
                cdl_i += 1
                if baseline_results is not None:
                    baseline_score = baseline_results[json_fname].score
                else:
                    baseline_score = None
                
                mutate = (ID, L, json_fname, baseline_score)
                yield codeling.Codeling(cfg, mutate=mutate, deferred=True)


# TODO XXX uses LastID, needs a rewrite
# also get rid of old 'if STOPPING'
def gtor_concat(cfg: 'config.Config', gen: int) -> 'codeling.Codeling':
    IDs = LastID(write_at_exit=False)
    alive = all_alive_json()
    for fn1 in alive:
        cdl1 = codeling.Codeling(fn1)
        gen1 = cdl1.gen()
        print(cdl1.json['ID'], '...')
        
        for fn2 in alive:
            if STOPPING:
                print(' Caught SIGINT, stopping. Waiting for jobs to finish.',
                      file=sys.stderr)
                break
            
            cdl2 = codeling.Codeling(fn2)
            gen2 = cdl2.gen()
            
            if gen in (gen1, gen2):
                child_gen = f"{ max(gen1, gen2) + 1 :04x}"
                child_ID = f"{child_gen}-{IDs.next_ID(child_gen):08x}"
                child = cdl1.concat(cdl2, child_ID)
                yield child


def family(cfg: 'config.Config', scores: list, n_accept: int):
    cdl_res = collections.namedtuple('cdl_res', 'cdl res')
    cdls = {}
    for json_fname, res in scores.items():
        cdl = codeling.Codeling(cfg, json_fname=json_fname)
        cdls[cdl.json['ID']] = cdl_res(cdl, res)
    
    def earliest_ancestors(cdl_ID: str):
        """earliest ancestor(s) still in cfg.indir"""
        if cdl_ID not in cdls:
            return None
        else:
            parents = cdls[cdl_ID].cdl.json['parents']
            eas = list(filter(None, map(earliest_ancestors, parents)))
            return set.union(*eas) if eas else {cdl_ID}
    
    fams = {}
    def family_ID(cdl_ID: str):
        for fam_ID, fam in fams.items():
            if cdl_ID in fam:
                return fam_ID
        
        return None
    
    # figure out families
    for cdl_ID in cdls:
        eas = earliest_ancestors(cdl_ID)
        fam_IDs = sorted(set(filter(None, map(family_ID, eas))))
        if not fam_IDs:
            # start a new family
            fam_ID = sorted(eas)[0]
            fams[fam_ID] = eas | {cdl_ID}
        else:
            # merge everyone into one big family
            fam_ID = fam_IDs.pop(0)
            fams[fam_ID] |= eas | {cdl_ID}
            for del_ID in fam_IDs:
                fams[fam_ID] |= fams[del_ID]
                del(fams[del_ID])
    
    print()
    IDlen = 20
    for fam_ID in sorted(fams):
        print(f"Family of {fam_ID}:")
        fam = sorted(fams[fam_ID],
                     key = lambda cID: (-cdls[cID].res.score, cID))
        n_accepted = 0
        for cdl_ID in fam:
            r = cdls[cdl_ID].res
            if n_accepted < n_accept:
                n_accepted += 1
                status = 'accept'
                cdls[cdl_ID].cdl.link_json_wasm(cfg.outdir)
            else:
                status = 'reject'
            
            print(f"{cdl_ID:{IDlen}}", f"{r.score:>7x}", f"{status:>7}",
                  r.desc, sep="\t")
        
        print()


def find_codeling(cfg: 'config.Config', json_or_ID: str):
    j = json_or_ID
    if not j.endswith('.json'):
        j += '.json'
    
    for d in ('', cfg.outdir, cfg.indir, cfg.histdir):
        f = os.path.join(d, j)
        if os.path.isfile(f):
            return f
    
    raise RuntimeError(f"cannot find JSON file or codeling ID '{json_or_ID}'")


def fetch_codelings(cfg: 'config.Config', json_or_IDs: list,
                    incl_parents: bool = False):
    cdls = {}
    
    def fetch_codeling(json_or_ID: str):
        nonlocal cfg, cdls, incl_parents
        
        if json_or_ID in cdls:
            return
        
        json_fname = find_codeling(cfg, json_or_ID)
        cdl = codeling.Codeling(cfg, json_fname=json_fname)
        ID = cdl.json['ID']
        if ID in cdls:
            return
        
        targ = 0
        gen = codelang.Function(parse=(cdl.b(), targ)).dump(tokens=True)
        cdl._f = [item for instr in gen for item in instr]
        cdls[ID] = cdl
        
        if incl_parents:
            for p in cdl.json['parents']:
                fetch_codeling(p)
    
    for i in json_or_IDs:
        if os.path.isdir(i):
            for j in glob.glob(os.path.join(i, '*.json')):
                fetch_codeling(j)
        else:
            fetch_codeling(i)
    
    return cdls


def dump(cfg: 'config.Config', json_or_IDs: list):
    cdls = fetch_codelings(cfg, json_or_IDs)
    prtr = codelang_diff.IndelPrinter(outformat=cfg.format)
    
    for _, cdl in sorted(cdls.items()):
        print(json.dumps(cdl.json, indent=3))
        codelang_diff.CodelangDiff(prtr, cdl._f, None)
    
    prtr.EOT()


def history(cfg: 'config.Config', json_or_IDs: list):
    cdls = fetch_codelings(cfg, json_or_IDs, incl_parents=True)
    prtr = codelang_diff.IndelPrinter(outformat=cfg.format)
    
    for _, cdl in sorted(cdls.items()):
        print(json.dumps(cdl.json, indent=3))
        len_parents = len(cdl.json['parents'])
        if len_parents == 0:
            codelang_diff.CodelangDiff(prtr, cdl._f, [])
        elif len_parents == 1:
            parent_f = cdls[cdl.json['parents'][0]]._f
            codelang_diff.CodelangDiff(prtr, cdl._f, parent_f)
        else:
            raise RuntimeError('can only handle up to 1 parent')
    
    prtr.EOT()


def LEB128_test():
    old = (0x00, 0x3f, 0x40, 0x7f, 0x80, 0xbf, 0xc0, 0x1fff, 0x2000,
           0x7ffffff, 0x8000000,
           0x7fffffff, 0x80000000, 0xf7ffffff, 0xf8000000,
           0xffffffbf, 0xffffffc0, 0xffffffff)
    
    # generated via:
    # bits = random.sample(list(range(1,32)), 16)
    # list(map(hex,sorted((random.getrandbits(i) for i in bits))))
    new = (0x0, 0x2, 0x19, 0xc7, 0x1f4, 0xbab, 0x1c39, 0x20ed, 
           0x13d2e, 0x16e9b, 0x2b86e, 0x631c2, 
           0x2dd1fb, 0x5b956a, 0x51c3180, 0xfd4d929)
    
    for i32 in new:
        b = util.LEB128(util.unsigned2signed(i32, 32))
        bs = util.ByteStream(b)
        parsed = util.signed2unsigned(bs.next_LEB128(), 32)
        assert parsed == i32
        print(f"i32={hex(i32)}  b=0x{b.hex()}  parsed={hex(parsed)}")


def hack():
    LEB128_test()


def print_item_desc(content):
    """The single argument `content` looks as follows:
    
        [('item1', 'long description of item 1'),
         ('item2', 'long description of item 2'), ...]
    
    Print this out in a nice way using textwrap.wrap(...).
    
    Used by list_available_xxxx() below.
    """
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
    for fn in dir(codeling.Codeling):
        m = re.match("score_(.*)", fn)
        if m:
            fn_code = m.group(1)
            doc = getattr(getattr(codeling.Codeling, fn), '__doc__')
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


def get_release():
    with open('release', 'r') as f:
        return f.readline().strip()


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
          # scoring function 'Codeling.score_v02()', print out their scores
          # and save those with score >= 0x00 to '{cfg.outdir}'
          {cmd} -rnd0 10 -length 5 -scfn v02 -thresh 0x00 run02
          
        Feel free to use Ctrl-C to gracefully end the script at any point.
        """
    
    defaults = (
        ('length', 5),
        ('fuel', 50),
        ('scfn', 'Lv2_nolen'),
        ('mtfn', 'ins'),
        ('thresh', None),
        ('nproc', multiprocessing.cpu_count()),
        ('format', 'ANSI'),
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
        if hasattr(codeling.Codeling, fn):
            return s
        else:
            raise argparse.ArgumentTypeError(f"there is no '{fn}' in Codeling")
    
    def type_mtfn(s: str):
        fn = 'mutator_' + s
        if hasattr(codelang.Function, fn):
            return s
        else:
            raise argparse.ArgumentTypeError(
                f"there is no '{fn}' in codelang.Function")
    
    def type_dir_str(s: str):
        if os.path.isdir(s):
            return s
        else:
            raise argparse.ArgumentTypeError(f"'{s}' is not a directory")
    
    parser = argparse.ArgumentParser(
        description=main.__doc__, epilog=textwrap.dedent(epilogue),
        formatter_class=argparse.RawDescriptionHelpFormatter)
    
    cmds = parser.add_mutually_exclusive_group(required=True)
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
        X+Y by concatenating their codes. XXX TODO: FIX, CURRENTLY BROKEN""")
    cmds.add_argument('-score', action='store_true',
        help=f"""Score all codelings in '{cfg.indir}'.""")
    cmds.add_argument('-family', type=type_int_ish, metavar='N',
        help=f"""The codelings in '{cfg.indir}' are grouped into families. Two 
        codelings in '{cfg.indir}' are in the same family when one is a parent 
        of the other. Family membership is transitive, i.e. if codelings A and 
        B are in the same family and codelings B and C are in the same family, 
        then provided that all three codelings are in '{cfg.indir}', all three 
        are in the same family. N highest-scoring codelings in each family are 
        saved to '{cfg.outdir}'. If two or more codelings in the same family 
        have equal scores, they are saved in alphabetical order of their IDs 
        up to a total of N codelings per family.""")
    cmds.add_argument('-dump', type=str, metavar='cdl', nargs='+', 
        help=f"""Print out a codeling (or several codelings) including its 
        parsed code. The parameter 'cdl' can be one of the following: a 
        codeling JSON file (in which case it must end in '.json'), a codeling 
        ID (in which case 'ID.json' is searched for in '{cfg.outdir}', 
        '{cfg.indir}', '{cfg.histdir}', in that order, until the file is 
        found), or a directory name (in which case all '.json' files in that 
        directory get dumped).""")
    cmds.add_argument('-history', type=str, metavar='cdl', nargs='+',
        help=f"""Similar to '-dump' above, but also prints out the whole 
        evolutionary history of each codeling, marking the changes in each 
        generation (green for insertions, red for deletions).""")
    
    parser.add_argument('-upto', action='store_true',
        help=f"""For options that use the length parameter L (see '-length' 
        below for details), equivalent to repeatedly running this script with 
        '-length 1', '-length 2', ..., '-length L'
        (e.g. '-upto -length 5').""")
    parser.add_argument('-length', type=type_int_ish, metavar='L',
        help=f"""For options that produce new codelings, set the 'length' 
        parameter to L. This parameter has slightly different meanings 
        depending on which method for generating or mutating codelings is 
        used: for -rnd0 it is the length of the new random strings in bytes, 
        for -gen0 it is the minimum number of instructions in the new 
        codelings, and for -mutate is the minimum number of instructions 
        changed in the parent to produce the new codelings.
        (Default: {cfg.length})""")
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
        mutator functions along with their descriptions.
        (Default: '{cfg.mtfn}')""")
    parser.add_argument('-diff', action='store_true',
        help=f"""The default setting is that the threshold T (see '-thresh' 
        below for details) is absolute, i.e. any codeling saved to 
        '{cfg.indir}' must have a score >= T. The '-diff' option changes this 
        to the difference between the codeling's own score and that of its 
        best-scoring parent, i.e. the test becomes score - best_parent >= 
        T.""")
    parser.add_argument('-thresh', type=type_int_ish, metavar='T',
        help=f"""Codelings with score >= T (e.g. '0x5f') are saved to 
        '{cfg.outdir}'. For existing codelings (e.g. those taken from 
        '{cfg.indir}') this creates hard links to the originals. If a negative 
        threshold is needed, use e.g. '" -0x40"' - note the quotation marks 
        and the initial space. (Default: {default_T})""")
    parser.add_argument('-keep-all', action='store_true',
        help=f"""The default setting is that newly generated codelings saved 
        to '{cfg.outdir}' are kept unique, i.e. they are only kept in 
        '{cfg.outdir}' if their code is different from other codelings already 
        in '{cfg.outdir}' as well as all the input codelings in '{cfg.indir}'. 
        When the '-keep-all' option is used, all newly generated codelings 
        above the threshold are saved to '{cfg.outdir}'.""")
    parser.add_argument('-indir', type=type_dir_str, metavar='path',
        help=f"""Change the input directory to 'path'.
        (Default: '{cfg.indir}')""")
    parser.add_argument('-outdir', type=type_dir_str, metavar='path',
        help=f"""Change the output directory to 'path'.
        (Default: '{cfg.outdir}')""")
    parser.add_argument('-nproc', type=type_int_ish, metavar='N', 
        help=f"""Set the number of worker processes to use in the pool to N. 
        (Default for this machine: {cfg.nproc})""")
    parser.add_argument('-format', choices=['ANSI', 'HTML'], 
        help=f"""Set the output format used by the '-history' option colour 
        mark-up to either the ANSI terminal escape sequences or simple HTML 
        with inline styles. Ignored when used with any other option. (Default: 
        {cfg.format})""")
    
    parser.add_argument('runid', type=str, nargs='?',
        help=f"""Identifier for this run (e.g. 'run01'). All new codelings 
        produced during the run will have identifiers of the form 
        'runid-123456789012', i.e. the run identifier followed by a dash and 
        twelve digits (with most of the leading digits being zero). The run 
        identifier is required for all runs where new codelings are produced 
        (i.e. '-rnd0', '-gen0', '-mutate' and '-concat').""")
    
    args = parser.parse_args()
    
    new_cdls = (args.rnd0, args.gen0, args.mutate, args.concat)
    if any(new_cdls) and args.runid is None:
        parser.error("'runid' is required for all runs where new codelings "
                     "are produced, i.e. '-rnd0', '-gen0', '-mutate' and "
                     "'-concat'")
    
    if args.runid is not None and re.match(r'^#', args.runid):
        parser.error("'runid' cannot start with '#'")
    
    if args.diff and not any((args.mutate, args.concat)):
        parser.error("'-diff' can only be used with '-mutate' and '-concat'")
    
    for param in ('length fuel scfn mtfn diff thresh keep_all '
                  'indir outdir nproc format runid').split():
        val = getattr(args, param)
        if val is not None or param == 'runid':
            setattr(cfg, param, val)
    
    if args.upto:
        Ls = range(1, cfg.length + 1)
    else:
        Ls = (cfg.length,)
    
    with open(cfg.template_file, "rb") as f:
        cfg.template = f.read()
    
    signal.signal(signal.SIGINT, SIGINT_handler)
    
    print('#', ' '.join(sys.argv))
    
    if args.dump:
        dump(cfg, args.dump)
        return
    elif args.history:
        history(cfg, args.history)
        return
    elif args.rnd0 is not None:
        sys.exit("SORRY, -rnd0 not implemented yet (well, re-implemented) :-(")
    elif args.score or args.family:
        # no new codelings are generated
        cfg.runid = None
        cfg.keep_all = True
    
    print('#')
    if args.family:
        print(f"# Scoring all codelings in '{cfg.indir}'")
        cfg.baseline_run = True
        baseline_results = score_Codelings(cfg, stop_check(gtor_score(cfg)))
        family(cfg, baseline_results, args.family)
        return
    elif cfg.diff:
        print(f"# Baseline scores for all codelings in '{cfg.indir}'")
        cfg.baseline_run = True
        baseline_results = score_Codelings(cfg, stop_check(gtor_score(cfg)))
        print('#')
        print('# Main run')
    else:
        baseline_results = None
    
    if args.score:
        gtor = gtor_score(cfg)
    elif args.gen0 is not None:
        gtor = gtor_gen0(cfg, Ls, args.gen0)
    elif args.mutate is not None:
        gtor = gtor_mutate(cfg, Ls, args.mutate, baseline_results)
    elif args.concat is not None:
        gtor = gtor_concat(cfg, args.concat)
    else:
        gtor = None
    
    cfg.baseline_run = False
    score_Codelings(cfg, stop_check(gtor))


if __name__ == "__main__":
    main()
