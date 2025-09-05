pub const contract = @import("contract/new.zig").new;
pub const reduce = @import("contract/new.zig").new;

test {
    @import("std").testing.refAllDecls(@This());
}
