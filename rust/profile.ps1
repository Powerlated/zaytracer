$uProfPath = "C:\Program Files\AMD\AMDuProf\bin\AMDuProfCLI.exe"
$uProfArgs = @("collect", "--config", "tbp", "-o", "uprof_out", "target/release/profiling.exe")

cargo build --workspace --release
if ($LASTEXITCODE -ne 0) { 
    throw "External command failed with exit code $LASTEXITCODE" 
}
& $uProfPath $uProfArgs 