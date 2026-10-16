const std = @import("std");
const Io = std.Io;

const builtin = @import("builtin");
const c = @import("c");

const sdl_log = std.log.scoped(.sdl);
const app_log = std.log.scoped(.app);

pub const std_options: std.Options = .{ .log_level = .debug };

const target_triple: [:0]const u8 = x: {
    var buf: [256]u8 = undefined;
    var fba: std.heap.FixedBufferAllocator = .init(&buf);
    break :x (builtin.target.zigTriple(fba.allocator()) catch unreachable) ++ "";
};

const zaytracer = @import("zaytracer");
const zaytracer_core = @import("zaytracer_core.zig");

// We will use this renderer to draw into this window every frame.
var window: *c.SDL_Window = undefined;
var renderer: *c.SDL_Renderer = undefined;
var texture: *c.SDL_Texture = undefined;
var texture_width: c_int = 0;
var texture_height: c_int = 0;
var surface: *c.SDL_Surface = undefined;

pub const WINDOW_WIDTH: c_int = 640;
pub const WINDOW_HEIGHT: c_int = 480;

fn sdlAppInit(appstate: ?*?*anyopaque, argv: [][*:0]u8) !c.SDL_AppResult {
    _ = appstate;
    _ = argv;
    _ = c.SDL_SetAppMetadata("Example Renderer Geometry", "1.0", "com.example.renderer-geometry");

    try errify(c.SDL_Init(c.SDL_INIT_VIDEO));
    try errify(c.SDL_CreateWindowAndRenderer("SDL in Zig", WINDOW_WIDTH, WINDOW_HEIGHT, 0, @ptrCast(&window), @ptrCast(&renderer)));
    _ = c.SDL_SetRenderLogicalPresentation(renderer, WINDOW_WIDTH, WINDOW_HEIGHT, c.SDL_LOGICAL_PRESENTATION_LETTERBOX);

    // Textures are pixel data that we upload to the video hardware for fast drawing. Lots of 2D
    // engines refer to these as "sprites." We'll do a static texture (upload once, draw many
    // times) with data from a bitmap file.

    surface = try errify(c.SDL_CreateSurface(WINDOW_WIDTH, WINDOW_HEIGHT, c.SDL_PIXELFORMAT_ARGB8888));

    texture_width = surface.w;
    texture_height = surface.h;

    texture = try errify(c.SDL_CreateTextureFromSurface(renderer, surface));

    return c.SDL_APP_CONTINUE; // carry on with the program!
}

// This function runs once per frame, and is the heart of the program.
fn sdlAppIterate(appstate: ?*anyopaque) !c.SDL_AppResult {
    _ = appstate;


    zaytracer_core.zaytrace_image(@ptrCast(@alignCast(surface.pixels)));

    try errify(c.SDL_RenderClear(renderer));
    
    try errify(c.SDL_UpdateTexture(texture, null, surface.pixels, surface.pitch));
    // null srcrect and dstrect mean the entire rendering target
    try errify(c.SDL_RenderTexture(renderer, texture, null, null));

    try errify(c.SDL_RenderPresent(renderer));

    return c.SDL_APP_CONTINUE;
}

// This function runs when a new event (mouse input, keypresses, etc) occurs.
fn sdlAppEvent(appstate: ?*anyopaque, event: *c.SDL_Event) !c.SDL_AppResult {
    _ = appstate;

    if (event.type == c.SDL_EVENT_QUIT) {
        return c.SDL_APP_SUCCESS; // end the program, reporting success to the OS.
    }
    return c.SDL_APP_CONTINUE; // carry on with the program!
}

fn sdlAppQuit(appstate: ?*anyopaque, result: anyerror!c.SDL_AppResult) void {
    _ = appstate;

    _ = result catch |err| if (err == error.SdlError) {
        sdl_log.err("{s}", .{c.SDL_GetError()});
    };
}

// https://github.com/castholm/zig-examples/blob/master/breakout/main.zig#L945

/// Converts the return value of an SDL function to an error union.
inline fn errify(value: anytype) error{SdlError}!switch (@typeInfo(@TypeOf(value))) {
    .bool => void,
    .pointer, .optional => @TypeOf(value.?),
    .int => |info| switch (info.signedness) {
        .signed => @TypeOf(@max(0, value)),
        .unsigned => @TypeOf(value),
    },
    else => @compileError("unerrifiable type: " ++ @typeName(@TypeOf(value))),
} {
    return switch (@typeInfo(@TypeOf(value))) {
        .bool => if (!value) error.SdlError,
        .pointer, .optional => value orelse error.SdlError,
        .int => |info| switch (info.signedness) {
            .signed => if (value >= 0) @max(0, value) else error.SdlError,
            .unsigned => if (value != 0) value else error.SdlError,
        },
        else => comptime unreachable,
    };
}

//#region SDL main callbacks boilerplate

pub var gpa: std.mem.Allocator = undefined;

pub fn main(init: std.process.Init) !u8 {
    // This is appropriate for anything that lives as long as the process.
    gpa = init.arena.allocator();

    var empty_argv: [0:null]?[*:0]u8 = .{};
    const status: u8 = @truncate(@as(c_uint, @bitCast(c.SDL_RunApp(empty_argv.len, @ptrCast(&empty_argv), sdlMainC, null))));
    return app_err.load() orelse status;
}

fn sdlMainC(argc: c_int, argv: ?[*:null]?[*:0]u8) callconv(.c) c_int {
    return c.SDL_EnterAppMainCallbacks(argc, @ptrCast(argv), sdlAppInitC, sdlAppIterateC, sdlAppEventC, sdlAppQuitC);
}

fn sdlAppInitC(appstate: ?*?*anyopaque, argc: c_int, argv: ?[*:null]?[*:0]u8) callconv(.c) c.SDL_AppResult {
    return sdlAppInit(appstate.?, @ptrCast(argv.?[0..@intCast(argc)])) catch |err| app_err.store(err);
}

fn sdlAppIterateC(appstate: ?*anyopaque) callconv(.c) c.SDL_AppResult {
    return sdlAppIterate(appstate) catch |err| app_err.store(err);
}

fn sdlAppEventC(appstate: ?*anyopaque, event: ?*c.SDL_Event) callconv(.c) c.SDL_AppResult {
    return sdlAppEvent(appstate, event.?) catch |err| app_err.store(err);
}

fn sdlAppQuitC(appstate: ?*anyopaque, result: c.SDL_AppResult) callconv(.c) void {
    sdlAppQuit(appstate, app_err.load() orelse result);
}

var app_err: ErrorStore = .{};

const ErrorStore = struct {
    const status_not_stored = 0;
    const status_storing = 1;
    const status_stored = 2;

    status: c.SDL_AtomicInt = .{ .value = status_not_stored },
    err: anyerror = undefined,
    trace_index: usize = undefined,
    trace_addrs: [32]usize = undefined,

    fn store(es: *ErrorStore, err: anyerror) c.SDL_AppResult {
        if (c.SDL_CompareAndSwapAtomicInt(&es.status, status_not_stored, status_storing)) {
            es.err = err;
            if (@errorReturnTrace()) |src_trace| {
                es.trace_index = src_trace.index;
                const len = @min(es.trace_addrs.len, src_trace.instruction_addresses.len);
                @memcpy(es.trace_addrs[0..len], src_trace.instruction_addresses[0..len]);
            }
            _ = c.SDL_SetAtomicInt(&es.status, status_stored);
        }
        return c.SDL_APP_FAILURE;
    }

    fn load(es: *ErrorStore) ?anyerror {
        if (c.SDL_GetAtomicInt(&es.status) != status_stored) return null;
        if (@errorReturnTrace()) |dst_trace| {
            dst_trace.index = es.trace_index;
            const len = @min(dst_trace.instruction_addresses.len, es.trace_addrs.len);
            @memcpy(dst_trace.instruction_addresses[0..len], es.trace_addrs[0..len]);
        }
        return es.err;
    }
};

//#endregion SDL main callbacks boilerplate

test {
    // Register zaytracer_core.zig for testing
    _ = @import("zaytracer_core.zig");
}

// pub fn orig_main(init: std.process.Init) !void {
    // // Prints to stderr, unbuffered, ignoring potential error.
    // std.debug.print("All your {s} are belong to us.\n", .{"codebase"});

    // // This is appropriate for anything that lives as long as the process.
    // // const arena: std.mem.Allocator = init.arena.allocator();

    // // Accessing command line arguments:
    // const args = try init.minimal.args.toSlice(arena);
    // for (args) |arg| {
    //     std.log.info("arg: {s}", .{arg});
    // }

    // // In order to do I/O operations need an `Io` instance.
    // const io = init.io;

    // // Stdout is for the actual output of your application, for example if you
    // // are implementing gzip, then only the compressed bytes should be sent to
    // // stdout, not any debugging messages.
    // var stdout_buffer: [1024]u8 = undefined;
    // var stdout_file_writer: Io.File.Writer = .init(.stdout(), io, &stdout_buffer);
    // const stdout_writer = &stdout_file_writer.interface;

    // try zaytracer.printAnotherMessage(stdout_writer);

    // try stdout_writer.flush(); // Don't forget to flush!
// }

// test "simple test" {
//     const gpa = std.testing.allocator;
//     var list: std.ArrayList(i32) = .empty;
//     defer list.deinit(gpa); // Try commenting this out and see if zig detects the memory leak!
//     try list.append(gpa, 42);
//     try std.testing.expectEqual(@as(i32, 42), list.pop());
// }

// test "fuzz example" {
//     try std.testing.fuzz({}, testOne, .{});
// }

// fn testOne(context: void, smith: *std.testing.Smith) !void {
//     _ = context;
//     // Try command `zig build test --fuzz -Doptimize=ReleaseFast` to see if it manages to fail this test case!

//     const gpa = std.testing.allocator;
//     var list: std.ArrayList(u8) = .empty;
//     defer list.deinit(gpa);
//     while (!smith.eos()) switch (smith.value(enum { add_data, dup_data })) {
//         .add_data => {
//             const slice = try list.addManyAsSlice(gpa, smith.value(u4));
//             smith.bytes(slice);
//         },
//         .dup_data => {
//             if (list.items.len == 0) continue;
//             if (list.items.len > std.math.maxInt(u32)) return error.SkipZigTest;
//             const len = smith.valueRangeAtMost(u32, 1, @min(32, list.items.len));
//             const off = smith.valueRangeAtMost(u32, 0, @intCast(list.items.len - len));
//             try list.appendSlice(gpa, list.items[off..][0..len]);
//             try std.testing.expectEqualSlices(
//                 u8,
//                 list.items[off..][0..len],
//                 list.items[list.items.len - len ..],
//             );
//         },
//     };
// }
