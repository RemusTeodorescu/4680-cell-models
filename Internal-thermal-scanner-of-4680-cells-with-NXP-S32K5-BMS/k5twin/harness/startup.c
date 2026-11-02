/* k5twin - minimal Cortex-M7 startup for the QEMU mps2-an500 machine.
 * Enables the FPU (FPv5-D16, as on the S32K566 M7), copies .data,
 * clears .bss and calls main(). Faults are reported via semihosting.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

extern uint32_t _sidata, _sdata, _edata, _sbss, _ebss, _estack;
extern int main(void);
extern void __libc_init_array(void);
extern void initialise_monitor_handles(void);

void Reset_Handler(void);
void Fault_Handler(void);
void Fault_Report(uint32_t *frame);
void Default_Handler(void);

__attribute__((section(".isr_vector"), used))
void (*const g_vectors[16])(void) = {
    (void (*)(void))(&_estack),
    Reset_Handler,
    Default_Handler,   /* NMI        */
    Fault_Handler,     /* HardFault  */
    Fault_Handler,     /* MemManage  */
    Fault_Handler,     /* BusFault   */
    Fault_Handler,     /* UsageFault */
    0, 0, 0, 0,
    Default_Handler,   /* SVC        */
    Default_Handler,   /* DebugMon   */
    0,
    Default_Handler,   /* PendSV     */
    Default_Handler,   /* SysTick    */
};

#define SCB_CPACR (*(volatile uint32_t *)0xE000ED88u)
#define SCB_CFSR  (*(volatile uint32_t *)0xE000ED28u)
#define SCB_HFSR  (*(volatile uint32_t *)0xE000ED2Cu)
#define SCB_MMFAR (*(volatile uint32_t *)0xE000ED34u)
#define SCB_BFAR  (*(volatile uint32_t *)0xE000ED38u)
#define SCB_CCR   (*(volatile uint32_t *)0xE000ED14u)

void Reset_Handler(void)
{
    SCB_CPACR |= (0xFu << 20);          /* CP10/CP11 full access: FPU on */
    SCB_CCR |= (1u << 4);               /* trap divide by zero, as a debug aid */
    __asm volatile("dsb\n isb");

    uint32_t *src = &_sidata, *dst = &_sdata;
    while (dst < &_edata) *dst++ = *src++;
    for (dst = &_sbss; dst < &_ebss; ) *dst++ = 0;

    initialise_monitor_handles();       /* semihosting stdio (rdimon) */
    __libc_init_array();
    exit(main());
}

__attribute__((naked)) void Fault_Handler(void)
{
    __asm volatile("tst lr, #4\n ite eq\n mrseq r0, msp\n mrsne r0, psp\n b Fault_Report");
}

__attribute__((used)) void Fault_Report(uint32_t *frame)
{
    printf("K5TWIN_FAULT CFSR=0x%08lx HFSR=0x%08lx MMFAR=0x%08lx BFAR=0x%08lx PC=0x%08lx LR=0x%08lx\n",
           (unsigned long)SCB_CFSR, (unsigned long)SCB_HFSR,
           (unsigned long)SCB_MMFAR, (unsigned long)SCB_BFAR,
           (unsigned long)frame[6], (unsigned long)frame[5]);
    exit(3);
}

void Default_Handler(void)
{
    printf("K5TWIN_FAULT unexpected exception\n");
    exit(4);
}

/* -nostartfiles drops crti/crtn; newlib still references these. */
void _init(void) {}
void _fini(void) {}
