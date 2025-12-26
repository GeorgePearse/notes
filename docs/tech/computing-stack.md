# The Computing Stack (Bottom to Top)

## 1. Hardware
- Physical components: CPU, RAM, storage, peripherals
- The laptop's actual silicon and circuits

## 2. Firmware/BIOS/UEFI
- Low-level software burned into chips
- Initializes hardware on boot
- Hands control to the bootloader

## 3. Bootloader (e.g., GRUB)
- Loads the kernel into memory
- Passes initial parameters

## 4. Kernel (e.g., Linux, XNU, NT)
- Core of the OS
- Manages: memory, processes, device drivers, filesystems
- Provides syscalls - the interface between hardware and userspace
- Runs in "kernel space" with full hardware access

## 5. Operating System (userspace)
- Init system (systemd, launchd)
- System services and daemons
- Shell, file managers, window server
- Standard libraries (libc, etc.)
- The kernel + these components = "the OS"

## 6. Applications
- Web browsers, editors, games
- Run in "user space" - isolated from kernel
- Request resources through syscalls
- Can't directly touch hardware

## Key Insight

The kernel is the **gatekeeper** between applications and hardware. Apps never talk to hardware directly - they ask the kernel, which decides if/how to fulfill the request.
