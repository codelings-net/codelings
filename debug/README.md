## Instructions for debugging WebAssembly

Stepping through the code is an excellent way of gaining a better understanding 
of what the codelings are doing.

The best tool right now (March 2021) is Chrome because it shows the WebAssembly 
values stack and, if you have access to version 89+, it also has a Memory 
Inspector.

If you follow the instructions in this file, you should be able achieve 
something close to what's shown in the following screenshots:

Chrome:

![Chrome](chrome-screenshot.png)

Firefox:

![Firefox](firefox-screenshot.png)


### Symlink the `wasm` file you want debug to `debug/debug.wasm`

If you do not have a `wasm` file you would like to debug, you can skip this 
step and use the sample `debug.wasm` file.

If you have a `wasm` file you want to debug:

```bash
cd debug
mv debug.wasm debug.wasm.bak
ln -s path/to/you/file.wasm debug.wasm
```

### Start the HTTP server

```bash
cd debug
./http_server.py
```

### Start the debugger

Go to the following URL:

http://localhost:4096/debug.html

Open developer tools:

- In the desktop version of Chrome, the shortcut is **Ctrl+Shift+I** or Menu 
(three vertical dots in the top right corner) > More Tools > Developer tools.

- In the desktop version of Firefox, the shortcut is **Ctrl+Shift+Z** or Menu 
(three dashes in the top right corner) > Web developer > Debugger.

Hard-reload the page (left-click on the Reload button while holding `Shift`).

Find the `wasm` file among the sources and click on it. You should now see 
some WebAssembly code.

Toggle a breakpoint on one of the lines by clicking on the hex address on the 
left.

Hard-reload the page.

The debugger should now be paused on the breakpoint and you can step through 
the program by pressing F11. You should be able to see the current values of 
the 16 local variables, most of them zero. In Chrome, you should also be able 
to see the WebAssembly values stack.


### Add the watch expression

The watch expression dumps a few key regions of WebAssembly memory. If you feel 
like customising the JavaScript code for `dump()`, it is in `debug.html`.

The expressions to use are as follows:

```javascript
dump(memory0)  // Google Chrome
dump($m)       // Firefox
```


### Launch the Memory Inspector

The following only works in Chrome (sadly no Firefox equivalent) and only in 
version 89+, which as of March 2021 was only available from the developer beta 
channel (which is publicly accessible without the need for any sign-ups).

The first step is to enable WebAssembly debugging via:

DevConsole > Settings (cogged wheel, top right) > Experiments => (tick) 
WebAssembly Debugging: ...

and then right-click on:

DevConsole > Scope (in the right pane) > Module > memories > $m Memory(1)

and choose 'Inspect memory'.