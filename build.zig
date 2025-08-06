const std = @import("std");

// Although this function looks imperative, note that its job is to
// declaratively construct a build graph that will be executed by an external
// runner.
pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const ziggurat = b.dependency("ziggurat", .{
        .target = target,
        .optimize = optimize,
    });

    const lib_mod = b.createModule(.{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });

    lib_mod.addImport("ziggurat", ziggurat.module("ziggurat"));

    const lib = b.addLibrary(.{
        .linkage = .static,
        .name = "zither",
        .root_module = lib_mod,
    });

    b.installArtifact(lib);

    const lib_unit_tests = b.addTest(.{
        .root_module = lib_mod,
    });

    const ziggurat_unit_tests = b.addTest(.{
        .root_module = ziggurat.module("ziggurat"),
    });

    const run_lib_unit_tests = b.addRunArtifact(lib_unit_tests);
    const run_ziggurat_unit_tests = b.addRunArtifact(ziggurat_unit_tests);

    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_lib_unit_tests.step);
    test_step.dependOn(&run_ziggurat_unit_tests.step);
}
