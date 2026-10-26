use std::env;
use std::fs;
use std::path::{Path, PathBuf};

// Tested with AMD uProf version 5.3.521.0 on Windows x64

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=amd_uprof_wrapper.h");

    let amd_uprof_path = Path::new("C:\\Program Files\\AMD\\AMDuProf\\");
    let lib_dir = amd_uprof_path.join("lib\\x64");
    let include_dir = amd_uprof_path.join("include");

    println!("cargo:rustc-link-search={}", lib_dir.display());
    println!("cargo:rustc-link-lib=AMDProfileController");
    println!("cargo:rustc-link-lib=shell32");

    let builder = bindgen::Builder::default()
        .header("amd_uprof_wrapper.h")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .clang_arg(format!("-I{}", include_dir.display()));

    let bindings = builder.generate().unwrap();

    let out_path = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR not set"));
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Could not write bindings!");
}
