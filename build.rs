// Bakes the NixOS Vulkan loader/ICD paths into RPATH for our binaries.
// Without this, libvulkan.so.1 is only findable inside `nix develop` (where the
// flake's shell hook sets LD_LIBRARY_PATH), so running ./target/release/* from
// outside the dev shell silently falls back to the CPU backend.
//
// We probe for the NixOS-specific paths and emit -rpath only for ones that
// actually exist on this machine, so the build stays portable to non-NixOS
// Linux distros.

use std::path::Path;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    if !cfg!(target_os = "linux") {
        return;
    }

    // Order matters: the runtime loader searches these in order. The system
    // path is where NixOS installs libvulkan.so.1; the opengl-driver path is
    // where ICDs (libvulkan_*.so / *.json) live.
    let candidates = [
        "/run/current-system/sw/lib",
        "/run/opengl-driver/lib",
    ];

    for path in candidates {
        if Path::new(path).exists() {
            // -bins applies to every binary target in this package (rust_fun, train).
            println!("cargo:rustc-link-arg-bins=-Wl,-rpath,{path}");
        }
    }
}
