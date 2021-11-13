(module
  (func $fun
    (i32.const 0x00000000) drop nop nop
    
    (i32.const 0x0000003f) drop
    (i32.const 0x00000040) drop nop nop
    
    (i32.const 0x0000007f) drop
    (i32.const 0x00000080) drop nop nop
    
    (i32.const 0x000000bf) drop
    (i32.const 0x000000c0) drop nop nop
    
    (i32.const 0x00001fff) drop
    (i32.const 0x00002000) drop nop nop
    
    (i32.const 0x7fffffff) drop
    (i32.const 0x80000000) drop nop nop
    
    (i32.const 0xf7ffffff) drop
    (i32.const 0xf8000000) drop nop nop
    
    (i32.const 0xffffffbf) drop
    (i32.const 0xffffffc0) drop nop nop
    
    (i32.const -0x41) drop
    (i32.const -0x40) drop nop nop
    
    (i32.const 0xffffffff) drop
  )
)
