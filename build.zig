const std = @import("std");

// Although this function looks imperative, note that its job is to
// declaratively construct a build graph that will be executed by an external
// runner.
pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});

    const mod = b.addModule("zither", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
    });

    const ziggurat = b.dependency("ziggurat", .{
        .target = target,
    });

    const duct = b.dependency("duct", .{
        .target = target,
    });

    mod.addImport("ziggurat", ziggurat.module("ziggurat"));
    mod.addImport("duct", duct.module("duct"));

    const mod_tests = b.addTest(.{
        .root_module = mod,
        .name = "zither test",
    });

    const ziggurat_tests = b.addTest(.{
        .root_module = ziggurat.module("ziggurat"),
    });

    const duct_tests = b.addTest(.{
        .root_module = duct.module("duct"),
    });

    const run_mod_tests = b.addRunArtifact(mod_tests);
    const run_ziggurat_tests = b.addRunArtifact(ziggurat_tests);
    const run_duct_tests = b.addRunArtifact(duct_tests);

    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_mod_tests.step);

    const dependecy_test_step = b.step("dependencies", "Run dependency unit tests");
    dependecy_test_step.dependOn(&run_ziggurat_tests.step);
    dependecy_test_step.dependOn(&run_duct_tests.step);

    test_step.dependOn(dependecy_test_step);
}
