pub const contract = @import("contract/set.zig");
pub const reduce = @import("contract/new.zig");

test {
    @import("std").testing.refAllDecls(@This());
}
