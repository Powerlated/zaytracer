const std = @import("std");
const c = @import("c");

fn win32AssertBool(success: c_int) void {
    std.debug.assert(success == 1);
}

pub fn isolateCoreForThisProcess(core_index: u6, logical_proc_per_core: usize) void {
    var core_mask_to_clear: u64 = ~(@as(u64, 1) << core_index);
    std.debug.assert(core_index < 64);
    std.debug.assert(logical_proc_per_core == 1 or logical_proc_per_core == 2);

    if (logical_proc_per_core == 2) {
        std.debug.assert(core_index < 32);
        var newCoreMask: u64 = 0;
        for (0..32) |i_usize| {
            const i: u6 = @intCast(i_usize);
            newCoreMask |= (((core_mask_to_clear >> i) & 1)) << (i * 2);
            newCoreMask |= (((core_mask_to_clear >> i) & 1) << 1) << (i * 2);
        }
        core_mask_to_clear = newCoreMask;
    }

    std.debug.print("core bit mask: {b}\n", .{core_mask_to_clear});

    // Take snapshot of all running processes
    const hSnapshot = c.CreateToolhelp32Snapshot(c.TH32CS_SNAPPROCESS, 0);
    std.debug.assert(hSnapshot != c.INVALID_HANDLE_VALUE);

    var pe = std.mem.zeroes(c.PROCESSENTRY32);
    pe.dwSize = @sizeOf(c.PROCESSENTRY32);

    win32AssertBool(c.Process32First(hSnapshot, &pe));

    var processes_isolated: u32 = 0;
    while (true) {
        const hProcess = c.OpenProcess(c.PROCESS_QUERY_INFORMATION | c.PROCESS_SET_INFORMATION, c.FALSE, pe.th32ProcessID);
        if (hProcess != null) {
            var process_affinity_mask: u64 = 0;
            var system_affinity_mask: u64 = 0;

            win32AssertBool(c.GetProcessAffinityMask(hProcess, &process_affinity_mask, &system_affinity_mask));

            const updated_process_mask: u64 = process_affinity_mask & core_mask_to_clear;
            if (updated_process_mask != 0) {
                win32AssertBool(c.SetProcessAffinityMask(hProcess, updated_process_mask));
                processes_isolated += 1;
            }
        }

        if (c.FALSE == c.Process32Next(hSnapshot, &pe)) {
            break;
        }
    }

    std.debug.print("{} processes successfully isolated from physical core {}\n", .{ processes_isolated, core_index });

    const hProcess = c.GetCurrentProcess();

    const this_process_affinity_mask: u64 = @as(u64, 1) << @intCast(core_index * logical_proc_per_core);
    win32AssertBool(c.SetProcessAffinityMask(hProcess, this_process_affinity_mask));
    std.debug.print("This process affinity successfully set to thread (1 of {}, 1-indexed) on physical core {}\n", .{ logical_proc_per_core, core_index });

    win32AssertBool(c.SetPriorityClass(hProcess, c.HIGH_PRIORITY_CLASS));
    std.debug.print("This process successfully set to high priority\n", .{});
}
