use std::process::Command;
use std::io::*;

fn main() {
    let output = Command::new("git")
        .args(["rev-parse", "--short=7", "HEAD"])
        .output()
        .unwrap();

    let git_hash_untrimmed = String::from_utf8(output.stdout).unwrap();
    let git_hash: &str = &git_hash_untrimmed.trim();

    let is_dirty = Command::new("git")
        .args(["status", "--porcelain"])
        .output().unwrap().stdout.len() > 0;

    let git_hash_and_dirty_str = format!("{}{}",
        git_hash,
        if is_dirty {"*"} else {""}
    );

    let out_dir = std::env::var("OUT_DIR").unwrap();
    let dest_path = std::path::Path::new(&out_dir).join("generated_data.rs");
    let mut file = std::fs::File::create(&dest_path).unwrap();

    writeln!(file,
        "pub const GIT_HASH_AND_DIRTY_STR: &'static str = \"{}\";",
        git_hash_and_dirty_str
    ).unwrap();
    
    // Re-run this build script if the Git head or index changes
    println!("cargo:rerun-if-changed=.git/HEAD");
    println!("cargo:rerun-if-changed=.git/index");
}