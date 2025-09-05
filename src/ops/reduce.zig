pub const new = @import("reduce/new.zig").new;
pub const set = @import("reduce/set.zig").set;

test {
    @import("std").testing.refAllDecls(@This());
}
