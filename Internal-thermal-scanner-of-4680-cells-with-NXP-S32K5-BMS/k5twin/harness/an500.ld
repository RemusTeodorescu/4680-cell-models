/* k5twin run layout for QEMU mps2-an500 (Cortex-M7).
 * The address map differs from the S32K566; sizes are checked separately
 * against the real S32K566 M7_0 regions by s32k566_m7_check.ld. */
ENTRY(Reset_Handler)
MEMORY
{
  CODE (rx)  : ORIGIN = 0x00000000, LENGTH = 4M
  RAM  (rwx) : ORIGIN = 0x20000000, LENGTH = 4M
}
_estack = ORIGIN(RAM) + LENGTH(RAM);
_stack_size = 0x10000;
SECTIONS
{
  .isr_vector : { KEEP(*(.isr_vector)) } > CODE
  .text : {
    *(.text*) *(.rodata*) *(GLOW_WEIGHTS)
    KEEP(*(.init)) KEEP(*(.fini))
    . = ALIGN(4);
    __preinit_array_start = .; KEEP(*(.preinit_array)) __preinit_array_end = .;
    __init_array_start = .;    KEEP(*(SORT(.init_array.*))) KEEP(*(.init_array)) __init_array_end = .;
    __fini_array_start = .;    KEEP(*(.fini_array)) __fini_array_end = .;
  } > CODE
  .ARM.extab : { *(.ARM.extab* .gnu.linkonce.armextab.*) } > CODE
  .ARM.exidx : { __exidx_start = .; *(.ARM.exidx* .gnu.linkonce.armexidx.*) __exidx_end = .; } > CODE
  _sidata = LOADADDR(.data);
  .data : { . = ALIGN(4); _sdata = .; *(.data*) . = ALIGN(4); _edata = .; } > RAM AT > CODE
  .bss (NOLOAD) : { . = ALIGN(4); _sbss = .; *(.bss*) *(COMMON) *(GLOW_IO) . = ALIGN(4); _ebss = .; } > RAM
  .heap (NOLOAD) : { . = ALIGN(8); end = .; _end = .; __end__ = .; } > RAM
  _stack_limit = _estack - _stack_size;
  ASSERT(_ebss <= _stack_limit, "k5twin: RAM overflow")
}
