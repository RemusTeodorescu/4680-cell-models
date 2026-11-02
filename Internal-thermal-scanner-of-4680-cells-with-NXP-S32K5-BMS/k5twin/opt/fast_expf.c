/* fast_expf.c - drop-in replacement for newlib's expf, for Glow tanh/sigmoid.
 *
 * Glow's float tanh kernel calls expf() once per element. newlib's
 * wf_exp/ef_exp (__ieee754_expf) is written for full IEEE edge-case
 * handling. This version does Cody-Waite range reduction and a degree-6
 * polynomial: max relative error ~2e-7 on [-87, 88] (about 2 float ulp),
 * no table, no division.
 *
 * Use on the board: add this file to the firmware sources (CMake
 * add_executable/target_sources). The linker then resolves the bundle's
 * expf reference to it instead of libm.a. Check the accuracy first with
 * k5twin (--link opt/fast_expf.c).
 */
#include <stdint.h>
#include <string.h>

float expf(float x)
{
    if (x > 88.72283f) return __builtin_inff();
    if (x < -87.33654f) return 0.0f;
    float t = x * 1.44269504f;                          /* x / ln2 */
    int k = (int)(t + (t >= 0.0f ? 0.5f : -0.5f));
    float fk = (float)k;
    float r = x - fk * 0.693359375f;                    /* Cody-Waite: ln2 = hi + lo */
    r = r - fk * -2.12194440e-4f;
    float p = 1.0f / 720.0f;
    p = p * r + 1.0f / 120.0f;
    p = p * r + 1.0f / 24.0f;
    p = p * r + 1.0f / 6.0f;
    p = p * r + 0.5f;
    p = p * r + 1.0f;
    p = p * r + 1.0f;
    uint32_t bits = (uint32_t)(k + 127) << 23;
    float scale;
    memcpy(&scale, &bits, sizeof scale);
    return p * scale;
}
