pub const new = @import("contract/new.zig").new;
pub const set = @import("contract/set.zig").set;

test {
    @import("std").testing.refAllDecls(@This());
}
