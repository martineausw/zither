pub const all = @import("ops/new.zig");
pub const new = @import("ops/all.zig");

test {
    @import("std").testing.refAllDecls(@This());
}
