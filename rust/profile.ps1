$uProfPath = "C:\Program Files\AMD\AMDuProf\bin\AMDuProfCLI.exe"

cargo build --workspace --release
if ($LASTEXITCODE -ne 0) { 
    throw "External command failed with exit code $LASTEXITCODE" 
}

# https://docs.amd.com/r/en-US/57368-uProf-user-guide/9.5.5.-Examples-Windows
$uProfCollectArgs = @("collect", "--start-paused", "-e", "cycles-not-in-halt", "-e", "retired-inst", "--interval", "1000000", "-o", "uprof_out", "target/release/profiling.exe")
& $uProfPath $uProfCollectArgs
if ($LASTEXITCODE -ne 0) { 
    throw "External command failed with exit code $LASTEXITCODE" 
}

& $uProfPath report -i uprof_out
if ($LASTEXITCODE -ne 0) { 
    throw "External command failed with exit code $LASTEXITCODE" 
}
