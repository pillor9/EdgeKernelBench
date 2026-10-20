---
id: rk3588_gpu/arch_facts
hardware: rk3588_gpu (Mali-G610 Valhall, OpenCL)
tags: [facts, memory, execution-width]
layers: []
evidence_basis: [vendor_doc, backend_contract]
version_sensitivity: mixed -- verify OpenCL driver and kernel limits for each deployment
---

# RK3588 GPU: architecture facts

Use the active OpenCL target for candidate legality and query the installed
runtime for language versions, extensions and device limits. A product brief
does not override the compiler or runtime. [leka-rk3588-contract]

## Product and architecture

- RK3588 combines four Cortex-A76 and four Cortex-A55 CPU cores with an
  Arm Mali-G610 MC4 GPU. [rockchip-rk3588-brief]
- Mali-G610 belongs to the Valhall architecture. Its hardware execution width
  does not by itself authorize OpenCL subgroup operations; the runtime must
  report the required language features and extensions. [arm-g610-counters]
- Mali uses a unified memory system. OpenCL local storage should not be
  treated as CUDA-style dedicated shared SRAM; evaluate whether reuse repays
  copies and synchronization. Contiguous, appropriately aligned access is a
  useful starting point. [arm-opencl-guide]
- Arm documents shader, cache, external-memory and queue counters for G610.
  Availability depends on the driver, kernel interfaces, permissions and
  collector. A documented counter is not evidence that it was collected.
  [arm-g610-counters]

## LEKA target and planning limits

- The target is `opencl -device=mali`. Plan within 1024 work-items per
  workgroup, 32,768 bytes of shared/local storage and 128 function arguments,
  and apply any stricter device or compiled-kernel limit.
  [leka-rk3588-contract]
- Query preferred workgroup multiples, per-axis maxima, local-memory size and
  type, OpenCL C version and extensions. A TVM `thread_warp_size` field is a
  compiler parameter, not proof of hardware width or subgroup support.
- OpenCL compilation uses the target device's driver. Do not transfer CUDA
  or Adreno-specific intrinsics, image rules or tuning constants to Mali.
- The default deep collector is disabled. Optional `devfreq` and `hwcpipe`
  results describe a device-wide window and remain `exact_bundle=false`.
  They cannot be assigned to an individual candidate or PrimFunc.
  [leka-rk3588-contract]
- Streamline, a compatible `gatord`, and libGPUCounters are possible sources
  of dynamic counters when correctly configured. Mali Offline Compiler
  supplies static estimates; it does not measure runtime latency.
  [arm-performance-studio] [arm-libgpucounters] [arm-malioc]

[rockchip-rk3588-brief]: https://www.rock-chips.com/uploads/pdf/2022.8.26/191/RK3588%20Brief%20Datasheet.pdf
[arm-g610-counters]: https://documentation-service.arm.com/static/64d494774a92140bb31d7ae5
[arm-opencl-guide]: https://developer.arm.com/-/media/developer/Graphics%20and%20Multimedia/Developer%20Guides%20-%20PDFs/Arm%20Guide%20to%20OpenCL%20Programming.pdf
[arm-performance-studio]: https://developer.arm.com/Tools%20and%20Software/Arm%20Performance%20Studio
[arm-libgpucounters]: https://github.com/ARM-software/libGPUCounters
[arm-malioc]: https://developer.arm.com/tools-and-software/mali-offline-compiler
[leka-rk3588-contract]: ../../sources/rk3588_gpu/backend_contract.json
