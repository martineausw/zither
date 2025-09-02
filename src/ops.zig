pub const contract = @import("ops/contract.zig").contract;
pub const each = @import("ops/each.zig");
pub const reduce = @import("ops/reduce.zig");

test {
    @import("std").testing.refAllDecls(@This());
}
