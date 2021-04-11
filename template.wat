(module
  (func $fun
    ;; write 0x12345678 to out
    (i32.store
      (i32.load (i32.const 0x0c))
      (i32.const 0x12345678)))
  
  (export "f" (func $fun))
  
  (memory (export "m") 1)
  (data (i32.const 0)
    ;; 1 wasm page = 64 kiB = addresses 0x0000 - 0xffff 
    ;;
    ;; 0x00 i32   start of rnd -> 0x00fc   (random bytes)
    ;; 0x04 i32   start of tmp -> 0x0100   (temporary data)
    ;; 0x08 i32   start of in  -> 0x8000   (input)
    ;; 0x0c i32   start of out -> 0xc000   (output)
    ;;
    ;; NB tmp-rnd = 4, i.e. one i32 value
    ;;
    "\fc\00\00\00\00\01\00\00\00\80\00\00\00\c0\00\00"))
