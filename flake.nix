{
  description = "artificial-stupidity: tiny from-scratch language model on top of tensor-ash's Vulkan GEMM";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs }:
    let
      systems = [ "x86_64-linux" "aarch64-linux" ];
      forAllSystems = nixpkgs.lib.genAttrs systems;
    in {
      devShells = forAllSystems (system:
        let pkgs = import nixpkgs { inherit system; }; in {
          default = pkgs.mkShell {
            packages = with pkgs; [
              cargo
              rustc
              rustfmt
              clippy
              shaderc
              vulkan-loader
              vulkan-tools
              vulkan-validation-layers
            ];

            shellHook = ''
              export LD_LIBRARY_PATH="${pkgs.vulkan-loader}/lib:/run/opengl-driver/lib:''${LD_LIBRARY_PATH:-}"
              echo "artificial-stupidity dev shell: Vulkan loader on LD_LIBRARY_PATH."
              echo "Run: cargo run --release"
            '';
          };
        });
    };
}
