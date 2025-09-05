pub const new = @import("all/new.zig");
pub const set = @import("all/set.zig");

test {
    @import("std").testing.refAllDecls(@This());
}
